/* mlvoice web client.
 *
 * No build step and no CDN: this is served by the same process as the API, so
 * a developer who can run the service can open the UI, and it works offline.
 *
 * Two decisions shape the whole file.
 *
 * Every clip is converted to WAV in the browser through
 * AudioContext.decodeAudioData before upload. One path covers a phone's m4a,
 * an mp3, and the WebM/Opus a MediaRecorder emits, and the server needs no
 * codec beyond libsndfile -- doing it server-side would mean shipping ffmpeg.
 *
 * The page asks the server what it needs (GET /v1/info) and hides the rest.
 * A deployment with no API keys shows no key field; one that does not require
 * consent shows no consent step and enrols the voice the moment a clip is
 * chosen. What is left is two inputs: a voice and some words.
 */
"use strict";

const state = {
  referenceWav: null,
  consentWav: null,
  challengeToken: null,
  voiceId: null,
  lastAudioUrl: null,
  /* Assume consent is required and auth is on until told otherwise. If the
   * info fetch fails, showing a field nobody needs is the harmless mistake;
   * skipping a step the server enforces is not. */
  consentRequired: true,
  authRequired: true,
  transcription: true,
};

const $ = (id) => document.getElementById(id);
const apiKey = () => $("apiKey").value.trim();

function setStatus(id, message, kind = "info") {
  const node = $(id);
  node.textContent = message;
  node.className = `status ${kind}`;
}

async function request(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  const key = apiKey();
  if (key) headers["X-API-Key"] = key;
  const response = await fetch(path, { ...options, headers });
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
  let offset = 44;
  for (let i = 0; i < samples; i += 1) {
    const clamped = Math.max(-1, Math.min(1, channelData[i]));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

async function toWav(arrayBuffer) {
  const context = new (window.AudioContext || window.webkitAudioContext)();
  try {
    const decoded = await context.decodeAudioData(arrayBuffer);
    /* Downmix rather than take channel 0: one side of a stereo interview
     * recording can be nearly silent. */
    const mono = new Float32Array(decoded.length);
    for (let channel = 0; channel < decoded.numberOfChannels; channel += 1) {
      const data = decoded.getChannelData(channel);
      for (let i = 0; i < decoded.length; i += 1) mono[i] += data[i] / decoded.numberOfChannels;
    }
    return { blob: encodeWav(mono, decoded.sampleRate), seconds: decoded.duration };
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
    return this.recorder !== null;
  }

  async start() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
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

/* ------------------------------------------------------------ the voice ---
 * Choosing a clip is the whole interaction. Transcribing it and enrolling it
 * are consequences, not steps the user has to know about -- except where the
 * server requires consent, which cannot be inferred and has to be asked for.
 */

async function useReference({ blob, seconds }) {
  state.referenceWav = blob;
  state.voiceId = null;
  $("voicePlayer").src = URL.createObjectURL(blob);
  $("voicePlayer").hidden = false;
  /* Shown from here until the voice is in: the automatic path covers the happy
   * case, and this is what is left when it cannot run -- no transcriber, a
   * transcript worth correcting, or consent still to record. */
  $("enrolRow").hidden = false;
  await transcribeReference({ thenEnrol: !state.consentRequired });
}

async function transcribeReference({ thenEnrol = false } = {}) {
  if (!state.transcription) {
    $("voiceAdvanced").open = true;
    setStatus(
      "voiceStatus",
      "This deployment cannot transcribe. Type what the clip says under Details, " +
        "then it is ready to use.",
      "warn",
    );
    return;
  }
  setStatus(
    "voiceStatus",
    "Listening to your clip… the first one on a fresh install downloads the " +
      "recogniser, which takes a few minutes.",
  );
  const form = new FormData();
  form.append("audio", state.referenceWav, "reference.wav");
  try {
    const response = await request("/v1/transcribe", { method: "POST", body: form });
    const body = await response.json();
    $("referenceText").value = body.text;
    setStatus("voiceStatus", `Heard ${body.duration_seconds.toFixed(1)}s of speech.`, "ok");
    if (thenEnrol) await enrol();
  } catch (error) {
    /* Recoverable: the transcript can be typed. Open the panel that holds the
     * field rather than leaving the user to find it. */
    $("voiceAdvanced").open = true;
    setStatus(
      "voiceStatus",
      `Could not transcribe (${error.message}). Type what the clip says under ` +
        "Details — the words in the recording, not the text you want generated.",
      "warn",
    );
  }
}

/* -------------------------------------------------------------- consent --- */

async function requestChallenge() {
  const name = $("subjectName").value.trim();
  if (!name) {
    setStatus("voiceStatus", "Enter the name of the person whose voice this is.", "warn");
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
    setStatus(
      "voiceStatus",
      "Read the sentence aloud and record it. The random words tie the recording " +
        "to this moment, so an old clip cannot be reused.",
    );
  } catch (error) {
    setStatus("voiceStatus", error.message, "error");
  }
}

/* ------------------------------------------------------------- enrolment --- */

function defaultVoiceName() {
  const typed = $("voiceName").value.trim();
  if (typed) return typed;
  const subject = $("subjectName").value.trim();
  if (subject) return subject;
  /* Something stable and human, so the picker is readable if several voices
   * accumulate on a shared deployment. */
  return `Voice ${new Date().toLocaleString()}`;
}

async function enrol() {
  if (!state.referenceWav) {
    setStatus("voiceStatus", "Choose or record a clip first.", "warn");
    return;
  }
  if (state.consentRequired && (!state.consentWav || !state.challengeToken)) {
    setStatus("voiceStatus", "Record the consent sentence first.", "warn");
    return;
  }
  const form = new FormData();
  form.append("name", defaultVoiceName());
  form.append("reference_audio", state.referenceWav, "reference.wav");
  const transcript = $("referenceText").value.trim();
  if (transcript) form.append("reference_text", transcript);
  /* Sent whenever present, even where the server does not require them, so a
   * deployment that turns consent back on keeps the records it collected. */
  if (state.challengeToken) form.append("consent_token", state.challengeToken);
  if (state.consentWav) form.append("consent_audio", state.consentWav, "consent.wav");

  setStatus("voiceStatus", "Adding the voice…");
  try {
    const response = await request("/v1/voices", { method: "POST", body: form });
    const voice = await response.json();
    state.voiceId = voice.id;
    $("enrolRow").hidden = true;
    setStatus("voiceStatus", "Voice ready. Type something below.", "ok");
    await loadVoices();
  } catch (error) {
    setStatus("voiceStatus", error.message, "error");
  }
}

async function loadVoices() {
  try {
    const response = await request("/v1/voices");
    const body = await response.json();
    const select = $("voiceSelect");
    select.replaceChildren();
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
  } catch (error) {
    /* Listing is a convenience; failing it must not interrupt the flow. */
  }
}

/* ---------------------------------------------------------------- speak --- */

/* Every value below came back from the server, but it started as text the user
 * typed, so it is written through textContent. An innerHTML path here would
 * execute markup pasted into the input box -- only in the pasting user's own
 * page, but that is still a scripting bug and there is no reason to have one. */
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
  summary.append(count, " chunk(s).");
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

function selectedVoiceId() {
  const select = $("voiceSelect");
  /* The picker wins when the user has touched it; otherwise the voice just
   * added, which is the one they are thinking about. */
  if (!$("voicesBlock").hidden && select.value) return select.value;
  return state.voiceId;
}

async function speak() {
  const text = $("speakText").value.trim();
  if (!text) {
    setStatus("speakStatus", "Type something to say.", "warn");
    return;
  }
  const started = performance.now();
  setStatus("speakStatus", "Generating… on CPU this takes a while.");
  $("speakButton").disabled = true;
  try {
    const response = await request("/v1/tts", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text, voice_id: selectedVoiceId() }),
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

async function acceptFile(file) {
  if (!file) return;
  setStatus("voiceStatus", `Reading ${file.name}…`);
  try {
    await useReference(await toWav(await file.arrayBuffer()));
  } catch (error) {
    setStatus("voiceStatus", `Could not read that file: ${error.message}`, "error");
  }
}

function bindDropTarget() {
  const drop = $("voiceDrop");
  ["dragenter", "dragover"].forEach((name) =>
    drop.addEventListener(name, (event) => {
      event.preventDefault();
      drop.classList.add("over");
    }),
  );
  ["dragleave", "drop"].forEach((name) =>
    drop.addEventListener(name, () => drop.classList.remove("over")),
  );
  drop.addEventListener("drop", (event) => {
    event.preventDefault();
    acceptFile(event.dataTransfer.files[0]);
  });
}

function applyCapabilities() {
  $("authCard").hidden = !state.authRequired;
  $("consentBlock").hidden = !state.consentRequired;
}

window.addEventListener("DOMContentLoaded", async () => {
  $("voiceFile").addEventListener("change", (event) => acceptFile(event.target.files[0]));
  bindDropTarget();
  bindRecorder(referenceRecorder, $("voiceRecord"), useReference, "voiceStatus");
  bindRecorder(
    consentRecorder,
    $("consentRecord"),
    ({ blob, seconds }) => {
      state.consentWav = blob;
      $("consentPlayer").src = URL.createObjectURL(blob);
      $("consentPlayer").hidden = false;
      setStatus("voiceStatus", `${seconds.toFixed(1)}s captured. Now add the voice.`, "ok");
    },
    "voiceStatus",
  );

  $("challengeButton").addEventListener("click", requestChallenge);
  $("enrolButton").addEventListener("click", enrol);
  $("analyseButton").addEventListener("click", analyse);
  $("speakButton").addEventListener("click", speak);
  $("retranscribe").addEventListener("click", () => {
    if (state.referenceWav) transcribeReference();
  });
  /* The voice list needs a key, and at first paint there is none. */
  $("apiKey").addEventListener("change", loadVoices);

  try {
    const info = await (await fetch("/v1/info")).json();
    state.consentRequired = info.consent_required !== false;
    state.authRequired = info.auth_required !== false;
    state.transcription = info.transcription !== false;
    applyCapabilities();
    $("backendInfo").textContent =
      `${info.backend.backend} · ${info.backend.model_id} · ${info.backend.device} · ` +
      `${info.environment}${info.watermarking ? " · watermarked" : ""}`;
    if (info.backend.backend === "dummy") {
      setStatus(
        "backendWarning",
        "This is the dummy backend: it is a test signal generator, not a model, " +
          "so anything you generate will be a buzz. Run `mlvoice doctor` to see " +
          "why, or set MLVOICE_TTS_BACKEND=indicf5.",
        "warn",
      );
    }
  } catch (error) {
    $("backendInfo").textContent = "service unreachable";
    applyCapabilities();
  }
  await loadVoices();
});
