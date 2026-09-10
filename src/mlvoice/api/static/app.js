/* mlvoice web client.
 *
 * No build step and no CDN: this is served by the same process as the API, so
 * a developer who can run the service can open the UI, and it works offline.
 *
 * The one non-obvious decision is that every clip is converted to WAV in the
 * browser via AudioContext.decodeAudioData before upload. That means the
 * server needs no codec beyond libsndfile while the user can hand over
 * whatever their phone produced -- m4a, mp3, or the webm/opus a MediaRecorder
 * emits. Doing it server-side would mean shipping ffmpeg.
 */
"use strict";

const state = {
  referenceWav: null,
  consentWav: null,
  challengeToken: null,
  voiceId: null,
  lastAudioUrl: null,
};

const $ = (id) => document.getElementById(id);
const apiKey = () => $("apiKey").value.trim();

function setStatus(id, message, kind = "info") {
  const node = $(id);
  node.textContent = message;
  node.className = `status ${kind}`;
}

function headers(extra = {}) {
  return { "X-API-Key": apiKey(), ...extra };
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: headers(options.headers || {}),
  });
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      detail = body.message || detail;
      if (body.context && body.context.reasons) {
        detail += `: ${body.context.reasons.join("; ")}`;
      }
      if (body.context && body.context.hint) detail += ` — ${body.context.hint}`;
    } catch (_) {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return response;
}

/* ---------------------------------------------------------------- audio ---
 * Decode with the browser, re-encode as 16-bit PCM WAV. One path for both
 * uploads and recordings, so the server sees a single format.
 */

function encodeWav(channelData, sampleRate) {
  const samples = channelData.length;
  const buffer = new ArrayBuffer(44 + samples * 2);
  const view = new DataView(buffer);
  const writeText = (offset, text) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  };
  writeText(0, "RIFF");
  view.setUint32(4, 36 + samples * 2, true);
  writeText(8, "WAVEfmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeText(36, "data");
  view.setUint32(40, samples * 2, true);
  for (let i = 0; i < samples; i += 1) {
    const clamped = Math.max(-1, Math.min(1, channelData[i]));
    view.setInt16(44 + i * 2, clamped * 32767, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

async function toWav(arrayBuffer) {
  const context = new (window.AudioContext || window.webkitAudioContext)();
  try {
    const decoded = await context.decodeAudioData(arrayBuffer);
    // Downmix to mono: the server would do it anyway, and it halves the upload.
    const length = decoded.length;
    const mono = new Float32Array(length);
    for (let channel = 0; channel < decoded.numberOfChannels; channel += 1) {
      const data = decoded.getChannelData(channel);
      for (let i = 0; i < length; i += 1) mono[i] += data[i] / decoded.numberOfChannels;
    }
    return {
      blob: encodeWav(mono, decoded.sampleRate),
      seconds: decoded.duration,
    };
  } finally {
    context.close();
  }
}

class Recorder {
  constructor() {
    this.recorder = null;
    this.chunks = [];
  }

  get active() {
    return this.recorder !== null && this.recorder.state === "recording";
  }

  async start() {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
    this.chunks = [];
    this.recorder = new MediaRecorder(stream);
    this.recorder.ondataavailable = (event) => {
      if (event.data.size > 0) this.chunks.push(event.data);
    };
    this.recorder.start();
  }

  async stop() {
    return new Promise((resolve) => {
      this.recorder.onstop = async () => {
        this.recorder.stream.getTracks().forEach((track) => track.stop());
        const blob = new Blob(this.chunks, { type: this.recorder.mimeType });
        this.recorder = null;
        resolve(await toWav(await blob.arrayBuffer()));
      };
      this.recorder.stop();
    });
  }
}

const referenceRecorder = new Recorder();
const consentRecorder = new Recorder();

/* ------------------------------------------------------------ reference --- */

async function useReference({ blob, seconds }) {
  state.referenceWav = blob;
  $("referencePlayer").src = URL.createObjectURL(blob);
  $("referencePlayer").hidden = false;
  setStatus("referenceStatus", `${seconds.toFixed(1)}s captured. Transcribing…`);
  await transcribeReference();
}

async function transcribeReference() {
  const form = new FormData();
  form.append("audio", state.referenceWav, "reference.wav");
  try {
    const response = await request("/v1/transcribe", { method: "POST", body: form });
    const body = await response.json();
    $("referenceText").value = body.text;
    const quality = body.quality || {};
    const bits = [
      `${body.duration_seconds.toFixed(1)}s`,
      `SNR ${Math.round(quality.estimated_snr_db)} dB`,
    ];
    setStatus(
      "referenceStatus",
      `${bits.join(" · ")} — transcript below, correct it if it is wrong.`,
      "ok",
    );
    const notes = $("referenceAdvisories");
    notes.innerHTML = "";
    (body.advisories || []).forEach((note) => {
      const item = document.createElement("li");
      item.textContent = note;
      notes.appendChild(item);
    });
    notes.hidden = (body.advisories || []).length === 0;
  } catch (error) {
    setStatus(
      "referenceStatus",
      `Could not transcribe (${error.message}). Type the transcript yourself below — ` +
        "it must be what this recording says, not the text you want generated.",
      "warn",
    );
  }
}

/* -------------------------------------------------------------- consent --- */

async function requestChallenge() {
  const name = $("subjectName").value.trim();
  if (!name) {
    setStatus("consentStatus", "Enter the name of the person whose voice this is.", "warn");
    return;
  }
  try {
    const response = await request("/v1/voices/challenge", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ subject_name: name }),
    });
    const body = await response.json();
    state.challengeToken = body.token;
    $("challengePhrase").textContent = body.phrase;
    $("challengeBlock").hidden = false;
    setStatus(
      "consentStatus",
      "Read the sentence below aloud and record it. The random words tie the " +
        "recording to this moment, so an old clip cannot be reused.",
    );
  } catch (error) {
    setStatus("consentStatus", error.message, "error");
  }
}

/* ------------------------------------------------------------- enrolment --- */

async function enrol() {
  if (!state.referenceWav) {
    setStatus("enrolStatus", "Add a reference recording first.", "warn");
    return;
  }
  if (!state.consentWav || !state.challengeToken) {
    setStatus("enrolStatus", "Record the consent sentence first.", "warn");
    return;
  }
  const form = new FormData();
  form.append("name", $("voiceName").value.trim() || $("subjectName").value.trim());
  form.append("consent_token", state.challengeToken);
  form.append("reference_audio", state.referenceWav, "reference.wav");
  form.append("consent_audio", state.consentWav, "consent.wav");
  const transcript = $("referenceText").value.trim();
  if (transcript) form.append("reference_text", transcript);

  setStatus("enrolStatus", "Enrolling…");
  try {
    const response = await request("/v1/voices", { method: "POST", body: form });
    const voice = await response.json();
    state.voiceId = voice.id;
    setStatus(
      "enrolStatus",
      `Enrolled as ${voice.id} (${voice.reference_duration_seconds.toFixed(1)}s reference, ` +
        `consent ${voice.consent_verified ? "verified" : "unverified"}).`,
      "ok",
    );
    await loadVoices();
    $("speakSection").hidden = false;
  } catch (error) {
    setStatus("enrolStatus", error.message, "error");
  }
}

async function loadVoices() {
  try {
    const response = await request("/v1/voices");
    const body = await response.json();
    const select = $("voiceSelect");
    select.innerHTML = "";
    /* An explicit empty option, so a voice can be deselected: without it the
     * first enrolled voice is implicitly selected and the model's own voice
     * becomes unreachable once anything is enrolled. */
    const base = document.createElement("option");
    base.value = "";
    base.textContent = "— the model's own voice —";
    select.appendChild(base);
    body.voices.forEach((voice) => {
      const option = document.createElement("option");
      option.value = voice.id;
      option.textContent = `${voice.name} — ${voice.status}`;
      option.selected = voice.id === state.voiceId;
      select.appendChild(option);
    });
    $("voicesBlock").hidden = body.voices.length === 0;
    if (body.voices.length > 0) $("speakSection").hidden = false;
  } catch (error) {
    /* listing failures are not worth interrupting the flow for */
  }
}

/* ---------------------------------------------------------------- speak --- */

/* Every value below comes back from the server, but it started as text the
 * user typed, so it is written through textContent rather than innerHTML. An
 * innerHTML path here would execute markup pasted into the input box -- only
 * in the pasting user's own page, but that is still a scripting bug and there
 * is no reason to have one. */
function renderAnalysis(body) {
  const tbody = $("analysisBody");
  tbody.replaceChildren();
  body.chunks.forEach((chunk) => {
    const row = tbody.insertRow();
    row.insertCell().textContent = String(chunk.index);
    row.insertCell().textContent = chunk.text;
    const phonemes = row.insertCell();
    phonemes.textContent = chunk.phonemes;
    phonemes.className = "mono";
    const brk = row.insertCell();
    brk.textContent = chunk.break_after;
    brk.appendChild(document.createElement("br"));
    const pause = document.createElement("small");
    pause.textContent = `${chunk.pause_ms}ms`;
    brk.appendChild(pause);
  });

  const summary = $("analysisSummary");
  summary.replaceChildren();
  const count = document.createElement("strong");
  count.textContent = String(body.chunks.length);
  summary.append(count, ` chunk(s).`);
  if (body.legacy_nta_fixed) {
    const fixed = document.createElement("strong");
    fixed.textContent = String(body.legacy_nta_fixed);
    summary.append(" Repaired ", fixed, " legacy ൻറ spelling(s).");
  }
  if (body.routed !== body.original) {
    const routed = document.createElement("span");
    routed.className = "ml";
    routed.textContent = body.routed;
    summary.append(
      document.createElement("br"),
      "What the model will actually be given:",
      document.createElement("br"),
      routed,
    );
  }
}

async function analyse() {
  const text = $("speakText").value.trim();
  if (!text) return;
  try {
    const response = await request("/v1/text/analyze", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text }),
    });
    renderAnalysis(await response.json());
  } catch (error) {
    $("analysisSummary").textContent = error.message;
    $("analysisBody").replaceChildren();
  }
  $("analysisBlock").hidden = false;
}

async function speak() {
  const text = $("speakText").value.trim();
  if (!text) {
    setStatus("speakStatus", "Type something to say.", "warn");
    return;
  }
  const voiceId = $("voiceSelect").value || null;
  const started = performance.now();
  setStatus("speakStatus", "Generating… on CPU this takes a while.", "info");
  $("speakButton").disabled = true;
  try {
    const response = await request("/v1/tts", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text, voice_id: voiceId }),
    });
    const blob = await response.blob();
    if (state.lastAudioUrl) URL.revokeObjectURL(state.lastAudioUrl);
    state.lastAudioUrl = URL.createObjectURL(blob);
    $("outputPlayer").src = state.lastAudioUrl;
    $("outputPlayer").hidden = false;
    const download = $("downloadLink");
    download.href = state.lastAudioUrl;
    download.hidden = false;

    const seconds = Number(response.headers.get("X-Audio-Duration") || 0);
    const rtf = Number(response.headers.get("X-Real-Time-Factor") || 0);
    const wall = (performance.now() - started) / 1000;
    setStatus(
      "speakStatus",
      `${seconds.toFixed(1)}s of audio in ${wall.toFixed(1)}s ` +
        `(real-time factor ${rtf.toFixed(2)}, ${response.headers.get("X-Chunks")} chunk(s)).`,
      "ok",
    );
  } catch (error) {
    setStatus("speakStatus", error.message, "error");
  } finally {
    $("speakButton").disabled = false;
  }
}

/* ----------------------------------------------------------------- wire --- */

function bindRecorder(recorder, button, onDone, statusId) {
  button.addEventListener("click", async () => {
    if (recorder.active) {
      button.textContent = button.dataset.idle;
      button.classList.remove("recording");
      onDone(await recorder.stop());
      return;
    }
    try {
      await recorder.start();
      button.dataset.idle = button.textContent;
      button.textContent = "■ Stop recording";
      button.classList.add("recording");
      setStatus(statusId, "Recording… speak now, then press stop.");
    } catch (error) {
      setStatus(statusId, `Microphone unavailable: ${error.message}`, "error");
    }
  });
}

function bindFileInput(input, onDone, statusId) {
  input.addEventListener("change", async () => {
    const file = input.files[0];
    if (!file) return;
    setStatus(statusId, `Converting ${file.name}…`);
    try {
      onDone(await toWav(await file.arrayBuffer()));
    } catch (error) {
      setStatus(statusId, `Could not read that file: ${error.message}`, "error");
    }
  });
}

window.addEventListener("DOMContentLoaded", async () => {
  bindFileInput($("referenceFile"), useReference, "referenceStatus");
  bindRecorder(referenceRecorder, $("referenceRecord"), useReference, "referenceStatus");

  bindRecorder(
    consentRecorder,
    $("consentRecord"),
    ({ blob, seconds }) => {
      state.consentWav = blob;
      $("consentPlayer").src = URL.createObjectURL(blob);
      $("consentPlayer").hidden = false;
      setStatus("consentStatus", `${seconds.toFixed(1)}s captured.`, "ok");
    },
    "consentStatus",
  );

  $("challengeButton").addEventListener("click", requestChallenge);
  $("enrolButton").addEventListener("click", enrol);
  $("analyseButton").addEventListener("click", analyse);
  $("speakButton").addEventListener("click", speak);
  /* The voice list needs a key, and at first paint there is none. Reload it
   * when one is entered, or an authenticated deployment shows an empty list
   * until something else happens to refresh it. */
  $("apiKey").addEventListener("change", loadVoices);
  $("retranscribe").addEventListener("click", () => {
    if (state.referenceWav) transcribeReference();
  });

  try {
    const info = await (await fetch("/v1/info")).json();
    $("backendInfo").textContent =
      `${info.backend.backend} · ${info.backend.model_id} · ${info.backend.device} · ` +
      `${info.environment}${info.watermarking ? " · watermarked" : ""}`;
    if (info.backend.backend === "dummy") {
      setStatus(
        "backendWarning",
        "The dummy backend is loaded: output will be a synthetic buzz, not speech. " +
          "Set MLVOICE_TTS_BACKEND=indicf5 for a real voice.",
        "warn",
      );
    }
  } catch (error) {
    $("backendInfo").textContent = "service unreachable";
  }
  await loadVoices();
});
