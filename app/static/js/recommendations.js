/**
 * SmartReco "For You" panel: polling, skeleton states and rendering.
 *
 * Behaviour:
 *   - Polls `GET /api/recommendations/latest` every 60 seconds.
 *   - Backs off to a faster 6-second poll while the agent is generating, so a
 *     freshly-triggered run appears almost immediately instead of up to a minute
 *     later.
 *   - Pauses entirely while the tab is hidden, and polls once on return. There is
 *     no reason to keep hitting the API for a page nobody is looking at.
 *   - Swaps between three mutually exclusive views: skeleton (generating),
 *     empty state, and the rendered recommendation.
 *   - Renders product cards with the same `data-track-*` attributes the server
 *     emits, so `tracker.js`'s delegated listener tracks them with no extra wiring.
 *
 * @module recommendations
 */
(function () {
  'use strict';

  /** @const {number} Normal poll cadence. */
  const POLL_INTERVAL_MS = 60000;

  /** @const {number} Faster cadence while a run is in flight. */
  const ACTIVE_POLL_INTERVAL_MS = 6000;

  const root = document.getElementById('for-you');
  if (!root || root.dataset.authenticated !== 'true') return;

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
    spinner: document.getElementById('reco-refresh-spinner'),
  };

  /** @type {?number} Currently rendered recommendation id. */
  let renderedId = root.dataset.hasRecommendation === 'true' ? -1 : null;
  /** @type {?number} Poll timer handle. */
  let timer = null;
  /** @type {number} Active poll cadence. */
  let cadence = root.dataset.pending ? ACTIVE_POLL_INTERVAL_MS : POLL_INTERVAL_MS;

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
  }

  /**
   * Tailwind classes for a skill-level badge.
   * @param {string} level
   * @return {string}
   */
  function levelClasses(level) {
    switch (String(level || '').toLowerCase()) {
      case 'beginner':
        return 'bg-sky-500/10 text-sky-300 ring-sky-500/25';
      case 'intermediate':
        return 'bg-amber-500/10 text-amber-300 ring-amber-500/25';
      case 'advanced':
        return 'bg-fuchsia-500/10 text-fuchsia-300 ring-fuchsia-500/25';
      default:
        return 'bg-ink-800 text-slate-400 ring-ink-600';
    }
  }

  // ---------------------------------------------------------------- rendering

  /**
   * Build the markup for one recommended product card.
   * Mirrors the `product_card` Jinja macro, including tracking attributes.
   * @param {Object} product
   * @param {number} recommendationId
   * @return {string}
   */
  function productCard(product, recommendationId) {
    const id = parseInt(product.id, 10);
    const trackAttrs =
      'data-track-click="' + id + '" ' +
      'data-track-source="recommendation" ' +
      'data-track-recommendation="' + recommendationId + '"';

    const media = product.thumbnail_url
      ? '<img src="' + escapeHtml(product.thumbnail_url) + '" alt="' + escapeHtml(product.title) +
        '" loading="lazy" class="h-36 w-full rounded-t-xl object-cover">'
      : '<div class="grid h-36 w-full place-items-center rounded-t-xl bg-gradient-to-br from-ink-800 via-ink-850 to-ink-900">' +
        '<span class="px-4 text-center font-mono text-[11px] uppercase tracking-[0.2em] text-slate-600">' +
        escapeHtml(product.category) + '</span></div>';

    const pitch = product.pitch || product.reason;
    const pitchBlock = pitch
      ? '<p class="rounded-lg border-l-2 border-signal-500/60 bg-signal-500/[0.06] px-3 py-2 text-[13px] italic leading-relaxed text-signal-100/90">' +
        escapeHtml(pitch) + '</p>'
      : '<p class="line-clamp-2 text-[13px] leading-relaxed text-slate-400">' +
        escapeHtml(product.description) + '</p>';

    const meta = [];
    if (product.duration) meta.push('<span>' + escapeHtml(product.duration) + '</span>');
    if (product.rating) {
      meta.push('<span class="text-amber-400/90">★ ' + parseFloat(product.rating).toFixed(1) + '</span>');
    }
    if (product.relevance_score !== null && product.relevance_score !== undefined) {
      meta.push('<span class="text-slate-600">rel ' + parseFloat(product.relevance_score).toFixed(2) + '</span>');
    }

    return (
      '<article class="group flex flex-col overflow-hidden rounded-xl border border-ink-700/70 bg-ink-900 transition hover:-translate-y-0.5 hover:border-ink-600 hover:shadow-xl hover:shadow-black/40">' +
        '<a href="/product/' + id + '" class="block" ' + trackAttrs + '>' + media + '</a>' +
        '<div class="flex flex-1 flex-col gap-3 p-4">' +
          '<div class="flex items-center justify-between gap-2">' +
            '<span class="truncate font-mono text-[10px] uppercase tracking-[0.15em] text-signal-400/80">' +
              escapeHtml(product.category) + '</span>' +
            '<span class="rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide ring-1 ring-inset ' +
              levelClasses(product.skill_level) + '">' + escapeHtml(product.skill_level) + '</span>' +
          '</div>' +
          '<h3 class="text-[15px] font-semibold leading-snug text-white">' +
            '<a href="/product/' + id + '" class="transition hover:text-signal-400" ' + trackAttrs + '>' +
              escapeHtml(product.title) + '</a>' +
          '</h3>' +
          pitchBlock +
          '<div class="mt-auto flex items-end justify-between gap-3 pt-1">' +
            '<div class="space-y-1">' +
              '<p class="text-base font-bold text-white">' + formatPrice(product.price) + '</p>' +
              '<p class="flex items-center gap-2 font-mono text-[10px] text-slate-500">' + meta.join('') + '</p>' +
            '</div>' +
            '<a href="/product/' + id + '" class="rounded-md border border-ink-600 px-3 py-1.5 text-xs font-semibold text-slate-200 transition group-hover:border-signal-500/50 group-hover:bg-signal-500/10 group-hover:text-signal-300" ' +
              trackAttrs + '>View</a>' +
          '</div>' +
        '</div>' +
      '</article>'
    );
  }

  /**
   * Build the markup for one interest signal row.
   * @param {Object} signal
   * @return {string}
   */
  function signalRow(signal) {
    const confidence = Math.round((parseFloat(signal.confidence) || 0) * 100);
    return (
      '<li class="rounded-lg border border-ink-700/70 bg-ink-900 p-3.5">' +
        '<div class="flex items-baseline justify-between gap-3">' +
          '<p class="text-sm font-semibold capitalize text-slate-100">' + escapeHtml(signal.topic) + '</p>' +
          '<p class="font-mono text-[11px] tabular-nums text-signal-400">' + confidence + '%</p>' +
        '</div>' +
        '<div class="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-ink-700" role="meter" aria-valuenow="' + confidence + '" aria-valuemin="0" aria-valuemax="100">' +
          '<div class="h-full rounded-full bg-gradient-to-r from-signal-600 to-signal-400" style="width: ' + confidence + '%"></div>' +
        '</div>' +
        (signal.evidence
          ? '<p class="mt-2.5 text-[13px] leading-relaxed text-slate-400">' + escapeHtml(signal.evidence) + '</p>'
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
      elements.trigger.textContent = 'trigger: ' + (recommendation.trigger_reason || 'manual');
    }
    if (elements.degraded) {
      elements.degraded.classList.toggle('hidden', !recommendation.degraded);
    }
    if (elements.eventCount) {
      elements.eventCount.textContent = recommendation.trigger_event_count || 0;
    }
    if (elements.latency) {
      elements.latency.textContent = recommendation.latency_ms
        ? Math.round(recommendation.latency_ms) + ' ms'
        : '—';
    }
    if (elements.created && recommendation.created_at) {
      const date = new Date(recommendation.created_at);
      elements.created.textContent = Number.isNaN(date.getTime())
        ? '—'
        : date.toLocaleString(undefined, {
            day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit',
          });
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

  /** Update the "checked at" stamp next to the refresh button. */
  function stampUpdated() {
    if (!elements.updated) return;
    const now = new Date();
    elements.updated.textContent =
      'checked ' + now.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
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
        if (payload.recommendation.id !== renderedId) render(payload.recommendation);
        return;
      }
      showView(payload.generating ? 'skeleton' : 'empty');
    } catch (error) {
      // Silent by design: a failed poll simply retries on the next tick.
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
      elements.refresh.disabled = true;
      if (elements.spinner) elements.spinner.classList.remove('hidden');
      if (renderedId === null) showView('skeleton');

      try {
        const response = await fetch('/api/recommendations/refresh', {
          method: 'POST',
          headers: { 'X-Requested-With': 'XMLHttpRequest' },
          credentials: 'same-origin',
        });

        if (response.status === 409) {
          // A run is already in flight — show the skeleton and let polling catch it.
          showView('skeleton');
        } else if (response.ok) {
          const payload = await response.json();
          if (payload.has_recommendation && payload.recommendation) {
            render(payload.recommendation);
          } else {
            showView('empty');
          }
        }
      } catch (error) {
        // Leave the current view in place; the next poll will reconcile it.
      } finally {
        elements.refresh.disabled = false;
        if (elements.spinner) elements.spinner.classList.add('hidden');
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

  restart();
  // Poll once shortly after load so a run triggered by this very page view
  // surfaces without waiting a full interval.
  window.setTimeout(poll, 2500);
})();
