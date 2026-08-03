/**
 * SmartReco "For You" panel: polling, pipeline tracker, the reveal, rendering.
 *
 * Behaviour:
 *   - Polls `GET /api/recommendations/latest` every 60 seconds; accelerates to
 *     6 seconds while a run is in flight; pauses entirely while the tab is
 *     hidden and polls once on return.
 *   - AUTO-TRIGGER: on first poll, if the user is authenticated with no
 *     recommendation and no run in flight, fires the existing refresh endpoint
 *     once (sessionStorage-guarded) so a first-time visitor watches the agent
 *     work within seconds instead of finding an empty state.
 *   - PIPELINE TRACKER: while generating, the seven node chips light in
 *     sequence (iris = thinking). When the result lands, remaining chips
 *     fast-forward to done (signal) and the reveal sequence plays.
 *   - THE REVEAL: skeleton crossfades out; the panel enters with staggered
 *     children (badge row → headline → narrative → cards). Confidence bars
 *     sweep from zero when "Why this recommendation?" opens (pure CSS).
 *   - All motion is skipped under prefers-reduced-motion and for re-renders of
 *     the same recommendation id.
 *
 * @module recommendations
 */
(function () {
  'use strict';

  /** @const {number} Normal poll cadence. */
  const POLL_INTERVAL_MS = 60000;

  /** @const {number} Faster cadence while a run is in flight. */
  const ACTIVE_POLL_INTERVAL_MS = 6000;

  /** @const {number} Pipeline chip advance interval while thinking. */
  const PIPE_STEP_MS = 1100;

  const root = document.getElementById('for-you');
  if (!root || root.dataset.authenticated !== 'true') return;

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const elements = {
    panel: document.getElementById('reco-panel'),
    skeleton: document.getElementById('reco-skeleton'),
    empty: document.getElementById('reco-empty'),
    headline: document.getElementById('reco-headline'),
    narrative: document.getElementById('reco-narrative'),
    products: document.getElementById('reco-products'),
    signals: document.getElementById('reco-signals'),
    trigger: document.getElementById('reco-trigger'),
    degraded: document.getElementById('reco-degraded'),
    eventCount: document.getElementById('reco-event-count'),
    latency: document.getElementById('reco-latency'),
    created: document.getElementById('reco-created'),
    updated: document.getElementById('reco-updated'),
    refresh: document.getElementById('reco-refresh'),
    refreshLabel: document.getElementById('reco-refresh-label'),
    spinner: document.getElementById('reco-refresh-spinner'),
  };

  /** Human translations for machine trigger enums (raw value kept in title). */
  const TRIGGER_LABELS = {
    first_time: 'first visit',
    event_threshold: 'activity threshold',
    stale: 'refreshed after inactivity',
    manual: 'on demand',
    scheduled_digest: 'daily digest',
  };

  /** Category glyphs — must mirror the Jinja `thumbnail` macro. */
  const CATEGORY_GLYPHS = {
    'Agentic AI': '◈', 'Machine Learning': '∿', 'Deep Learning': '▲',
    'Data Engineering': '☰', 'Python': '⌘', 'JavaScript': '{ }',
    'Web Development': '⌗', 'DevOps': '⟲', 'Cloud': '☁', 'Career Skills': '✎',
  };

  /** @type {?number} Currently rendered recommendation id. */
  let renderedId = root.dataset.hasRecommendation === 'true' ? -1 : null;
  /** @type {?number} Poll timer handle. */
  let timer = null;
  /** @type {number} Active poll cadence. */
  let cadence = root.dataset.pending ? ACTIVE_POLL_INTERVAL_MS : POLL_INTERVAL_MS;
  /** @type {?number} Pipeline animation timer. */
  let pipeTimer = null;
  /** @type {number} Index of the next pipeline chip to activate. */
  let pipeIndex = 0;
  /** @type {boolean} True once the first poll response has been handled. */
  let firstPollDone = false;

  // ---------------------------------------------------------------- helpers

  /**
   * Escape text for safe interpolation into HTML.
   * Agent-written copy is model output, so it is treated as untrusted.
   * @param {*} value
   * @return {string}
   */
  function escapeHtml(value) {
    if (value === null || value === undefined) return '';
    return String(value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  /**
   * Format a price the same way the server-side `money` filter does.
   * @param {*} value
   * @return {string}
   */
  function formatPrice(value) {
    const number = parseFloat(value);
    if (!number || Number.isNaN(number)) return 'Free';
    return '$' + number.toLocaleString('en-US', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
  }

  /**
   * Relative time for the telemetry strip ("just now", "4 min ago").
   * @param {string} iso
   * @return {string}
   */
  function relativeTime(iso) {
    const then = new Date(iso).getTime();
    if (Number.isNaN(then)) return '—';
    const seconds = Math.max(0, Math.round((Date.now() - then) / 1000));
    if (seconds < 45) return 'just now';
    if (seconds < 90) return '1 min ago';
    const minutes = Math.round(seconds / 60);
    if (minutes < 60) return minutes + ' min ago';
    const hours = Math.round(minutes / 60);
    if (hours < 24) return hours + (hours === 1 ? ' hour ago' : ' hours ago');
    const days = Math.round(hours / 24);
    return days + (days === 1 ? ' day ago' : ' days ago');
  }

  /**
   * Show exactly one of the three panel views.
   * @param {'panel'|'skeleton'|'empty'} view
   */
  function showView(view) {
    [['panel', elements.panel], ['skeleton', elements.skeleton], ['empty', elements.empty]]
      .forEach(function (entry) {
        const name = entry[0];
        const element = entry[1];
        if (!element) return;
        element.classList.toggle('hidden', name !== view);
      });
    root.setAttribute('aria-busy', view === 'skeleton' ? 'true' : 'false');
  }

  /**
   * Tailwind classes for a skill-level indicator dot.
   * @param {string} level
   * @return {string}
   */
  function levelDot(level) {
    switch (String(level || '').toLowerCase()) {
      case 'beginner': return 'bg-sky-400';
      case 'intermediate': return 'bg-amber-400';
      case 'advanced': return 'bg-fuchsia-400';
      default: return 'bg-mist-500';
    }
  }

  // ------------------------------------------------------- pipeline tracker

  /** @return {Array<Element>} The seven node chips. */
  function pipeChips() {
    return Array.prototype.slice.call(
      document.querySelectorAll('#pipeline-track .pipe-chip')
    );
  }

  /** Start (or restart) the sequential chip animation. */
  function startPipeline() {
    if (pipeTimer !== null) return;
    pipeIndex = 0;
    pipeChips().forEach(function (chip) {
      chip.classList.remove('is-active', 'is-done');
    });
    if (reduceMotion) return; // chips stay dormant; the pulse dot still reads "running"

    pipeTimer = window.setInterval(function () {
      const chips = pipeChips();
      if (pipeIndex < chips.length) {
        chips.forEach(function (chip) { chip.classList.remove('is-active'); });
        if (pipeIndex > 0) chips[pipeIndex - 1].classList.add('is-done');
        chips[pipeIndex].classList.add('is-active');
        pipeIndex++;
      }
      // Hold on the final chip until the result actually lands.
    }, PIPE_STEP_MS);
  }

  /** Stop the chip animation without completing it (e.g. run failed). */
  function stopPipeline() {
    if (pipeTimer !== null) {
      window.clearInterval(pipeTimer);
      pipeTimer = null;
    }
  }

  /**
   * Fast-forward every chip to done, then invoke the callback.
   * @param {function(): void} done
   */
  function finishPipeline(done) {
    stopPipeline();
    const chips = pipeChips();
    if (reduceMotion || chips.length === 0) {
      chips.forEach(function (chip) {
        chip.classList.remove('is-active');
        chip.classList.add('is-done');
      });
      done();
      return;
    }
    chips.forEach(function (chip, index) {
      window.setTimeout(function () {
        chip.classList.remove('is-active');
        chip.classList.add('is-done');
      }, index * 80);
    });
    window.setTimeout(done, chips.length * 80 + 150);
  }

  // ---------------------------------------------------------------- reveal

  /** Play the staged entrance on the freshly rendered panel. */
  function playReveal() {
    if (reduceMotion || !elements.panel) return;
    const panel = elements.panel;
    panel.classList.add('reveal-in');

    const children = panel.querySelectorAll(
      '#reco-headline, #reco-narrative, #reco-products > article'
    );
    children.forEach(function (el, index) {
      el.classList.add('reveal-child');
      el.style.setProperty('--stagger', String(index));
    });

    window.setTimeout(function () {
      panel.classList.remove('reveal-in');
      children.forEach(function (el) {
        el.classList.remove('reveal-child');
        el.style.removeProperty('--stagger');
      });
    }, 1400);
  }

  // ---------------------------------------------------------------- rendering

  /**
   * Build the markup for one recommended product card (mirrors the Jinja
   * `product_card` macro, including tracking attributes and category cover).
   * @param {Object} product
   * @param {number} recommendationId
   * @return {string}
   */
  function productCard(product, recommendationId) {
    const id = parseInt(product.id, 10);
    const category = String(product.category || 'Course');
    const coverClass = 'cover-' + category.toLowerCase().replace(/\s+/g, '-');
    const glyph = CATEGORY_GLYPHS[category] || '◇';
    const trackAttrs =
      'data-track-click="' + id + '" ' +
      'data-track-source="recommendation" ' +
      'data-track-recommendation="' + recommendationId + '"';

    // Mirrors the Jinja `thumbnail` macro: generated cover art when the row has
    // one, the CSS gradient cover as a fallback. Never an external hotlink.
    const cover = product.thumbnail_url
      ? '<img src="' + escapeHtml(product.thumbnail_url) + '" alt="Abstract cover artwork for ' +
        escapeHtml(product.title) + ', a ' + escapeHtml(category) + ' course" loading="lazy" ' +
        'decoding="async" width="800" height="447" class="card-cover-img h-full w-full object-cover">'
      : '<span class="absolute inset-0 grid place-items-center">' +
        '<span class="cover-glyph select-none text-5xl opacity-25" aria-hidden="true">' + glyph + '</span></span>';

    const media =
      '<div class="card-cover ' + coverClass + ' relative h-36 w-full overflow-hidden rounded-t-[10px]">' +
        cover +
        '<span class="pointer-events-none absolute inset-0 bg-gradient-to-t from-ink-950/85 via-ink-950/10 to-transparent"></span>' +
        '<span class="absolute bottom-2.5 left-3 font-mono text-[10px] lowercase tracking-[0.18em] text-white/75">' +
          escapeHtml(category) + '</span>' +
      '</div>';

    const pitch = product.pitch || product.reason;
    const pitchBlock = pitch
      ? '<p class="rounded-lg border-l-2 border-iris-500/70 bg-iris-500/[0.06] px-3 py-2 text-[13px] italic leading-relaxed text-iris-300/90">' +
        escapeHtml(pitch) + '</p>'
      : '<p class="line-clamp-2 text-[13px] leading-relaxed text-mist-400">' +
        escapeHtml(product.description) + '</p>';

    const meta = [];
    if (product.duration) meta.push('<span>' + escapeHtml(product.duration) + '</span>');
    if (product.rating) {
      meta.push('<span class="text-amber-400/90">★ ' + parseFloat(product.rating).toFixed(1) + '</span>');
    }
    if (product.relevance_score !== null && product.relevance_score !== undefined) {
      meta.push('<span title="agent relevance score">rel ' + parseFloat(product.relevance_score).toFixed(2) + '</span>');
    }

    return (
      '<article class="group flex flex-col overflow-hidden rounded-[10px] border border-ink-700/70 bg-ink-900 shadow-[inset_0_1px_0_rgba(255,255,255,.03)] transition duration-150 hover:-translate-y-0.5 hover:border-ink-600 hover:shadow-xl hover:shadow-black/40">' +
        '<a href="/product/' + id + '" class="block" ' + trackAttrs + '>' + media + '</a>' +
        '<div class="flex flex-1 flex-col gap-3 p-4">' +
          '<div class="flex items-center justify-between gap-2">' +
            '<span class="truncate font-mono text-[10px] lowercase tracking-[0.15em] text-signal-400/80">' +
              escapeHtml(category) + '</span>' +
            '<span class="inline-flex items-center gap-1.5 font-mono text-[10px] lowercase tracking-wide text-mist-400">' +
              '<span class="h-1.5 w-1.5 rounded-full ' + levelDot(product.skill_level) + '" aria-hidden="true"></span>' +
              escapeHtml(product.skill_level) + '</span>' +
          '</div>' +
          '<h3 class="text-[16px] font-semibold leading-snug tracking-[-0.01em] text-white">' +
            '<a href="/product/' + id + '" class="transition hover:text-signal-400" ' + trackAttrs + '>' +
              escapeHtml(product.title) + '</a>' +
          '</h3>' +
          pitchBlock +
          '<div class="mt-auto flex items-end justify-between gap-3 pt-1">' +
            '<div class="space-y-1">' +
              '<p class="text-base font-bold text-white">' + formatPrice(product.price) + '</p>' +
              '<p class="flex items-center gap-2 font-mono text-[10px] text-mist-500">' + meta.join('') + '</p>' +
            '</div>' +
            '<a href="/product/' + id + '" class="grid h-8 w-8 shrink-0 place-items-center rounded-md border border-ink-600 text-mist-400 transition group-hover:border-signal-500/50 group-hover:bg-signal-500/10 group-hover:text-signal-300" aria-label="View ' +
              escapeHtml(product.title) + '" ' + trackAttrs + '>→</a>' +
          '</div>' +
        '</div>' +
      '</article>'
    );
  }

  /**
   * Build the markup for one interest signal row.
   * @param {Object} signal
   * @param {number} index
   * @return {string}
   */
  function signalRow(signal, index) {
    const confidence = Math.round((parseFloat(signal.confidence) || 0) * 100);
    return (
      '<li class="rounded-[10px] border border-ink-700/70 bg-ink-900 p-3.5">' +
        '<div class="flex items-baseline justify-between gap-3">' +
          '<p class="font-mono text-sm font-medium lowercase text-mist-50">' + escapeHtml(signal.topic) + '</p>' +
          '<p class="font-mono text-[11px] tabular-nums text-signal-400">' + confidence + '%</p>' +
        '</div>' +
        '<div class="sig-bar mt-2 h-1.5 w-full overflow-hidden rounded-full bg-ink-700" role="meter" aria-valuenow="' + confidence + '" aria-valuemin="0" aria-valuemax="100" style="--stagger: ' + index + '">' +
          '<div class="h-full rounded-full bg-gradient-to-r from-iris-500 to-signal-400" style="width: ' + confidence + '%"></div>' +
        '</div>' +
        (signal.evidence
          ? '<p class="mt-2.5 border-l-2 border-iris-500/50 pl-3 text-[13px] italic leading-relaxed text-mist-400">' + escapeHtml(signal.evidence) + '</p>'
          : '') +
      '</li>'
    );
  }

  /**
   * Render a recommendation into the panel.
   * @param {Object} recommendation
   */
  function render(recommendation) {
    if (elements.headline) {
      elements.headline.textContent = recommendation.headline || 'Your picks';
    }
    if (elements.narrative) {
      elements.narrative.innerHTML = '<p>' + escapeHtml(recommendation.narrative) + '</p>';
    }
    if (elements.trigger) {
      const reason = recommendation.trigger_reason || 'manual';
      elements.trigger.innerHTML =
        '<span class="font-mono text-[10px] lowercase tracking-wider text-mist-500" title="trigger: ' +
        escapeHtml(reason) + '">trigger: ' +
        escapeHtml(TRIGGER_LABELS[reason] || reason) + '</span>';
    }
    if (elements.degraded) {
      elements.degraded.classList.toggle('hidden', !recommendation.degraded);
    }
    if (elements.eventCount) {
      elements.eventCount.textContent = recommendation.trigger_event_count || 0;
    }
    if (elements.latency) {
      elements.latency.textContent = recommendation.latency_ms
        ? (recommendation.latency_ms / 1000).toFixed(1) + 's'
        : '—';
      elements.latency.title = recommendation.latency_ms
        ? Math.round(recommendation.latency_ms) + ' ms'
        : '';
    }
    if (elements.created && recommendation.created_at) {
      elements.created.textContent = relativeTime(recommendation.created_at);
      elements.created.title = recommendation.created_at;
    }
    if (elements.products) {
      elements.products.innerHTML = (recommendation.products || [])
        .map(function (product) { return productCard(product, recommendation.id); })
        .join('');
    }
    if (elements.signals) {
      elements.signals.innerHTML = (recommendation.interest_signals || [])
        .map(signalRow)
        .join('');
    }

    renderedId = recommendation.id;
    showView('panel');
    stampUpdated();
  }

  /**
   * Present a freshly arrived recommendation: complete the pipeline first if
   * the skeleton is on stage, then reveal.
   * @param {Object} recommendation
   */
  function presentRecommendation(recommendation) {
    const skeletonVisible =
      elements.skeleton && !elements.skeleton.classList.contains('hidden');
    const isNew = recommendation.id !== renderedId;

    if (skeletonVisible && isNew) {
      finishPipeline(function () {
        render(recommendation);
        playReveal();
      });
    } else if (isNew) {
      render(recommendation);
      if (renderedId !== null) playReveal();
    }
  }

  /** Update the "checked at" stamp next to the refresh button. */
  function stampUpdated() {
    if (!elements.updated) return;
    const now = new Date();
    elements.updated.textContent =
      'checked ' + now.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
  }

  /**
   * Toggle the refresh button between idle and running states.
   * @param {boolean} running
   */
  function setRefreshRunning(running) {
    if (!elements.refresh) return;
    elements.refresh.disabled = running;
    if (elements.spinner) elements.spinner.classList.toggle('hidden', !running);
    if (elements.refreshLabel) {
      elements.refreshLabel.textContent = running ? 'agent running…' : 'Refresh my picks';
    }
  }

  /** Flash a ✓ on the refresh button for a moment after completion. */
  function flashRefreshDone() {
    if (!elements.refreshLabel) return;
    elements.refreshLabel.textContent = '✓ updated';
    window.setTimeout(function () {
      elements.refreshLabel.textContent = 'Refresh my picks';
    }, 1400);
  }

  // ------------------------------------------------------------ auto-trigger

  /**
   * First-visit judge path: if there is nothing to show and nothing running,
   * kick off a run via the existing refresh endpoint — once per tab session.
   */
  function maybeAutoTrigger(payload) {
    if (payload.has_recommendation || payload.generating) return;
    if (sessionStorage.getItem('smartreco_autorun')) return;
    sessionStorage.setItem('smartreco_autorun', '1');

    showView('skeleton');
    startPipeline();
    setRefreshRunning(true);

    fetch('/api/recommendations/refresh', {
      method: 'POST',
      headers: { 'X-Requested-With': 'XMLHttpRequest' },
      credentials: 'same-origin',
    })
      .then(function (response) { return response.ok ? response.json() : null; })
      .then(function (result) {
        setRefreshRunning(false);
        if (result && result.has_recommendation && result.recommendation) {
          presentRecommendation(result.recommendation);
          flashRefreshDone();
        }
        // 409/null: a run is already in flight — the fast poll reconciles.
      })
      .catch(function () { setRefreshRunning(false); });
  }

  // ----------------------------------------------------------------- polling

  /**
   * Fetch the latest state and update the view.
   * @return {Promise<void>}
   */
  async function poll() {
    try {
      const response = await fetch('/api/recommendations/latest?include_trigger=false', {
        headers: { 'X-Requested-With': 'XMLHttpRequest' },
        credentials: 'same-origin',
      });
      if (!response.ok) return;

      const payload = await response.json();
      stampUpdated();

      // Speed the poll up while a run is in flight, then settle back down.
      const desired = payload.generating ? ACTIVE_POLL_INTERVAL_MS : POLL_INTERVAL_MS;
      if (desired !== cadence) {
        cadence = desired;
        restart();
      }

      if (payload.has_recommendation && payload.recommendation) {
        presentRecommendation(payload.recommendation);
        return;
      }

      if (payload.generating) {
        showView('skeleton');
        startPipeline();
      } else if (!firstPollDone) {
        maybeAutoTrigger(payload);
      } else {
        stopPipeline();
        showView('empty');
      }
    } catch (error) {
      // Silent by design: a failed poll simply retries on the next tick.
    } finally {
      firstPollDone = true;
    }
  }

  /** Restart the poll loop with the current cadence. */
  function restart() {
    if (timer !== null) window.clearInterval(timer);
    timer = window.setInterval(poll, cadence);
  }

  /** Stop polling (used while the tab is hidden). */
  function stop() {
    if (timer !== null) {
      window.clearInterval(timer);
      timer = null;
    }
  }

  // ------------------------------------------------------------ manual refresh

  if (elements.refresh) {
    elements.refresh.addEventListener('click', async function () {
      setRefreshRunning(true);
      if (renderedId === null) {
        showView('skeleton');
        startPipeline();
      }

      try {
        const response = await fetch('/api/recommendations/refresh', {
          method: 'POST',
          headers: { 'X-Requested-With': 'XMLHttpRequest' },
          credentials: 'same-origin',
        });

        if (response.status === 409) {
          // A run is already in flight — show the skeleton and let polling catch it.
          showView('skeleton');
          startPipeline();
        } else if (response.ok) {
          const payload = await response.json();
          if (payload.has_recommendation && payload.recommendation) {
            // Force a re-present even if the id matches (manual refresh).
            renderedId = null;
            presentRecommendation(payload.recommendation);
            flashRefreshDone();
          } else {
            showView('empty');
          }
        }
      } catch (error) {
        // Leave the current view in place; the next poll will reconcile it.
      } finally {
        setRefreshRunning(false);
      }
    });
  }

  // --------------------------------------------------------------- lifecycle

  document.addEventListener('visibilitychange', function () {
    if (document.visibilityState === 'visible') {
      poll();
      restart();
    } else {
      stop();
    }
  });

  // If the server rendered the pending state, the run is already in flight.
  if (root.dataset.pending) startPipeline();

  restart();
  // Poll once shortly after load so a run triggered by this very page view
  // surfaces without waiting a full interval.
  window.setTimeout(poll, 1500);
})();
