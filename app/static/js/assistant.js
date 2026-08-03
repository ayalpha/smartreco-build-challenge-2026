/**
 * Nexora Assistant — personalised agent chat.
 *
 * Talks to two additive endpoints:
 *   GET  /api/assistant/profile  → the user's live interest signals (header chips)
 *   POST /api/assistant/chat     → a reply grounded in those same signals
 *
 * The personalisation is server-side: this module only renders it. No message
 * history is invented client-side and no answer is fabricated when the network
 * fails — errors surface as a visible, retryable state.
 *
 * @module assistant
 */
(function () {
  'use strict';

  const root = document.getElementById('assistant-root');
  if (!root) return;

  const el = {
    toggle: document.getElementById('assistant-toggle'),
    panel: document.getElementById('assistant-panel'),
    close: document.getElementById('assistant-close'),
    log: document.getElementById('assistant-log'),
    form: document.getElementById('assistant-form'),
    input: document.getElementById('assistant-input'),
    send: document.getElementById('assistant-send'),
    signals: document.getElementById('assistant-signals'),
    signalCount: document.getElementById('assistant-signal-count'),
    suggestions: document.getElementById('assistant-suggestions'),
  };

  let open = false;
  let busy = false;
  let profileLoaded = false;

  // ---------------------------------------------------------------- helpers

  /**
   * Escape untrusted text before it reaches innerHTML.
   * @param {*} value
   * @return {string}
   */
  function esc(value) {
    if (value === null || value === undefined) return '';
    return String(value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  /** Scroll the transcript to the newest message. */
  function scrollLog() {
    el.log.scrollTop = el.log.scrollHeight;
  }

  /**
   * Append a message bubble.
   * @param {'user'|'bot'|'error'} role
   * @param {string} text
   * @return {Element} the appended wrapper
   */
  function addMessage(role, text) {
    const wrap = document.createElement('div');
    wrap.className = 'assistant-msg assistant-msg--' + (role === 'user' ? 'user' : 'bot');

    const bubble = document.createElement('p');
    bubble.className = 'assistant-bubble assistant-bubble--' +
      (role === 'user' ? 'user' : role === 'error' ? 'error' : 'bot');
    bubble.textContent = text;

    wrap.appendChild(bubble);
    el.log.appendChild(wrap);
    scrollLog();
    return wrap;
  }

  /**
   * Append the course cards the assistant referenced.
   * @param {Array<Object>} courses
   */
  function addCourses(courses) {
    if (!courses || !courses.length) return;
    const wrap = document.createElement('div');
    wrap.className = 'assistant-msg assistant-msg--bot';

    wrap.innerHTML = '<div class="mt-1 w-full space-y-1.5">' + courses.map(function (c) {
      const cover = c.thumbnail_url
        ? '<img src="' + esc(c.thumbnail_url) + '" alt="" aria-hidden="true" loading="lazy" ' +
          'class="h-9 w-14 shrink-0 rounded object-cover">'
        : '';
      return '<a href="/product/' + parseInt(c.id, 10) + '" ' +
        'data-track-click="' + parseInt(c.id, 10) + '" data-track-source="assistant" ' +
        'class="flex items-center gap-2.5 rounded-lg border border-ink-700 bg-ink-900 p-2 transition hover:border-iris-500/50 hover:bg-ink-850">' +
        cover +
        '<span class="min-w-0 flex-1">' +
          '<span class="block truncate text-[12.5px] font-medium text-mist-100">' + esc(c.title) + '</span>' +
          '<span class="block font-mono text-[10px] lowercase text-mist-500">' +
            esc(c.category || '') + (c.skill_level ? ' · ' + esc(c.skill_level) : '') +
          '</span>' +
        '</span>' +
        '<span class="shrink-0 text-mist-500" aria-hidden="true">→</span>' +
        '</a>';
    }).join('') + '</div>';

    el.log.appendChild(wrap);
    scrollLog();
  }

  /**
   * Show the typing indicator.
   * @return {Element} the indicator node, to be removed when the reply lands
   */
  function addTyping() {
    const wrap = document.createElement('div');
    wrap.className = 'assistant-msg assistant-msg--bot';
    wrap.setAttribute('data-typing', '1');
    wrap.innerHTML =
      '<div class="assistant-bubble assistant-bubble--bot assistant-typing" aria-label="The assistant is thinking">' +
        '<span></span><span></span><span></span>' +
      '</div>';
    el.log.appendChild(wrap);
    scrollLog();
    return wrap;
  }

  // ---------------------------------------------------------------- profile

  /** Load and render the user's live interest signals into the header. */
  async function loadProfile() {
    if (profileLoaded) return;
    profileLoaded = true;
    try {
      const res = await fetch('/api/assistant/profile', {
        headers: { 'X-Requested-With': 'XMLHttpRequest' },
        credentials: 'same-origin',
      });
      if (!res.ok) return;
      const data = await res.json();

      const signals = data.signals || [];
      if (signals.length) {
        el.signals.innerHTML = signals.map(function (s) {
          const pct = Math.round((parseFloat(s.confidence) || 0) * 100);
          return '<li class="rounded-full border border-iris-500/30 bg-iris-500/10 px-2 py-0.5 ' +
            'font-mono text-[9.5px] lowercase text-iris-300">' + esc(s.topic) + ' ' + pct + '%</li>';
        }).join('');
        if (el.signalCount) {
          el.signalCount.textContent = signals.length + ' of your';
        }
      } else {
        el.signals.innerHTML =
          '<li class="font-mono text-[10px] lowercase text-mist-500">' +
          'no signals yet — browse a few courses and I sharpen up</li>';
      }
    } catch (error) {
      // Header chips are decorative; a failure must not block the chat.
    }
  }

  // ------------------------------------------------------------------- send

  /**
   * Send a message and render the grounded reply.
   * @param {string} text
   */
  async function send(text) {
    const message = String(text || '').trim();
    if (!message || busy) return;

    busy = true;
    el.send.disabled = true;
    el.input.value = '';
    el.input.style.height = 'auto';
    if (el.suggestions) el.suggestions.classList.add('hidden');

    addMessage('user', message);
    const typing = addTyping();

    try {
      const res = await fetch('/api/assistant/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'XMLHttpRequest' },
        credentials: 'same-origin',
        body: JSON.stringify({ message: message }),
      });

      typing.remove();

      if (res.status === 401) {
        addMessage('error', 'Your session expired. Please sign in again to keep chatting.');
        return;
      }
      if (!res.ok) {
        addMessage('error', 'I could not reach the agent just then. Try again in a moment.');
        return;
      }

      const data = await res.json();
      addMessage('bot', data.reply);
      addCourses(data.courses);

      if (data.degraded) {
        const note = document.createElement('p');
        note.className = 'px-1 font-mono text-[9.5px] lowercase text-mist-500';
        note.textContent = 'heuristic mode — model gateway unreachable for this reply';
        el.log.appendChild(note);
        scrollLog();
      }
    } catch (error) {
      typing.remove();
      addMessage('error', 'Network problem — your message was not sent. Try again.');
    } finally {
      busy = false;
      el.send.disabled = false;
      el.input.focus();
    }
  }

  // -------------------------------------------------------------- panel I/O

  /** Open the panel. */
  function openPanel() {
    open = true;
    el.panel.hidden = false;
    // Force a reflow so the entrance transition runs from the hidden state.
    void el.panel.offsetWidth;
    el.panel.classList.add('is-open');
    el.toggle.setAttribute('aria-expanded', 'true');
    el.toggle.classList.add('opacity-0', 'pointer-events-none');
    loadProfile();
    window.setTimeout(function () { el.input.focus(); }, 120);
  }

  /** Close the panel and return focus to the launcher. */
  function closePanel() {
    open = false;
    el.panel.classList.remove('is-open');
    el.toggle.setAttribute('aria-expanded', 'false');
    el.toggle.classList.remove('opacity-0', 'pointer-events-none');
    window.setTimeout(function () { if (!open) el.panel.hidden = true; }, 220);
    el.toggle.focus();
  }

  el.toggle.addEventListener('click', openPanel);
  el.close.addEventListener('click', closePanel);

  document.addEventListener('keydown', function (event) {
    if (event.key === 'Escape' && open) closePanel();
  });

  el.form.addEventListener('submit', function (event) {
    event.preventDefault();
    send(el.input.value);
  });

  // Enter sends, Shift+Enter newlines.
  el.input.addEventListener('keydown', function (event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      send(el.input.value);
    }
  });

  // Auto-grow the composer up to its max height.
  el.input.addEventListener('input', function () {
    el.input.style.height = 'auto';
    el.input.style.height = Math.min(el.input.scrollHeight, 112) + 'px';
  });

  // Suggested openers.
  root.querySelectorAll('.assistant-chip').forEach(function (chip) {
    chip.addEventListener('click', function () { send(chip.textContent); });
  });
})();
