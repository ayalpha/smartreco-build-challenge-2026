/**
 * SmartReco behavioural event tracker.
 *
 * Design rules (all enforced below):
 *   1. Never block the UI. Every send is fire-and-forget; a failure re-queues the
 *      batch and retries on the next flush rather than surfacing an error.
 *   2. Batch aggressively. Events are buffered and flushed every 5 seconds, so a
 *      burst of clicks costs one request, not ten.
 *   3. Survive page unload. `visibilitychange` + `pagehide` + `beforeunload` all
 *      flush via `navigator.sendBeacon`, which the browser delivers even after
 *      the document is gone. (`visibilitychange` is the reliable one on mobile,
 *      where `beforeunload` frequently never fires.)
 *   4. Measure real attention. Dwell time excludes periods when the tab was
 *      hidden, and is reported once per page as a `time_spent` event.
 *
 * Tracked events: page_view, product_click, search_query, time_spent,
 * add_to_cart, recommendation_click.
 *
 * @module tracker
 */
(function () {
  'use strict';

  /** @const {string} Endpoint that accepts batched events. */
  const ENDPOINT = '/api/events';

  /** @const {number} Flush cadence in milliseconds. */
  const FLUSH_INTERVAL_MS = 5000;

  /** @const {number} Maximum events held in memory before a forced flush. */
  const MAX_QUEUE_SIZE = 60;

  /** @const {number} Minimum dwell time worth reporting, in seconds. */
  const MIN_DWELL_SECONDS = 2;

  /** @const {string} sessionStorage key holding the session id. */
  const SESSION_KEY = 'smartreco_session_id';

  /**
   * Buffered, non-blocking behavioural event tracker.
   */
  class SmartRecoTracker {
    constructor() {
      /** @type {Array<Object>} Pending events awaiting flush. */
      this.queue = [];
      /** @type {string} Stable id for this browsing session. */
      this.sessionId = this.getOrCreateSession();
      /** @type {number} Accumulated visible time on this page, in ms. */
      this.visibleMs = 0;
      /** @type {?number} Timestamp when the page last became visible. */
      this.visibleSince = document.visibilityState === 'visible' ? Date.now() : null;
      /** @type {boolean} Guards against double-reporting dwell time. */
      this.dwellReported = false;
      /** @type {?number} Interval handle for the flush loop. */
      this.timer = null;

      this.productId = this.metaInt('smartreco-product-id');
      this.productTitle = this.meta('smartreco-product-title');
      this.path = this.meta('smartreco-path') || window.location.pathname;

      this.startFlushLoop();
      this.attachUnloadHandlers();
      this.attachVisibilityHandler();
      this.attachClickDelegation();
      this.trackInitialPageView();
      this.trackSearchIfPresent();
    }

    // ----------------------------------------------------------- utilities

    /**
     * Read a `<meta name=...>` content value.
     * @param {string} name
     * @return {?string}
     */
    meta(name) {
      const element = document.querySelector(`meta[name="${name}"]`);
      const value = element ? element.getAttribute('content') : null;
      return value && value.length ? value : null;
    }

    /**
     * Read a `<meta>` value as an integer.
     * @param {string} name
     * @return {?number}
     */
    metaInt(name) {
      const raw = this.meta(name);
      if (raw === null) return null;
      const parsed = parseInt(raw, 10);
      return Number.isNaN(parsed) ? null : parsed;
    }

    /**
     * Return the session id, generating and persisting one on first visit.
     * `sessionStorage` is per-tab and cleared when the tab closes, which is
     * exactly the lifetime a "session" should have.
     * @return {string}
     */
    getOrCreateSession() {
      try {
        const existing = window.sessionStorage.getItem(SESSION_KEY);
        if (existing) return existing;
        const generated = this.randomId();
        window.sessionStorage.setItem(SESSION_KEY, generated);
        return generated;
      } catch (error) {
        // Private browsing modes can throw on storage access.
        return this.randomId();
      }
    }

    /**
     * Generate a random session identifier.
     * @return {string}
     */
    randomId() {
      if (window.crypto && typeof window.crypto.randomUUID === 'function') {
        return window.crypto.randomUUID();
      }
      return 's-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 10);
    }

    // -------------------------------------------------------------- queueing

    /**
     * Queue an event for the next flush.
     * @param {string} eventType One of the six tracked event types.
     * @param {Object=} data Optional `product_id`, `path` and `metadata`.
     */
    track(eventType, data) {
      const payload = data || {};
      const event = {
        event_type: eventType,
        timestamp: new Date().toISOString(),
        path: payload.path || this.path,
        metadata: payload.metadata || {},
      };
      if (payload.product_id !== undefined && payload.product_id !== null) {
        event.product_id = payload.product_id;
      }

      this.queue.push(event);

      if (this.queue.length >= MAX_QUEUE_SIZE) {
        this.flush();
      }
    }

    /**
     * Send queued events. Re-queues the batch on failure so nothing is lost.
     * @return {Promise<void>}
     */
    async flush() {
      if (this.queue.length === 0) return;

      const batch = this.queue.splice(0);
      const body = JSON.stringify({ session_id: this.sessionId, events: batch });

      try {
        await fetch(ENDPOINT, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: body,
          keepalive: true,
          credentials: 'same-origin',
        });
      } catch (error) {
        // Put the events back at the front so ordering is preserved.
        this.queue.unshift.apply(this.queue, batch);
      }
    }

    /**
     * Flush synchronously via `sendBeacon` — the only reliable way to send
     * during unload. Falls back to `fetch(keepalive)` where unsupported.
     */
    flushBeacon() {
      if (this.queue.length === 0) return;

      const batch = this.queue.splice(0);
      const body = JSON.stringify({ session_id: this.sessionId, events: batch });

      if (navigator.sendBeacon) {
        // A Blob with an explicit JSON type keeps the server's content handling
        // simple; the endpoint accepts text/plain too, as a belt-and-braces measure.
        const blob = new Blob([body], { type: 'application/json' });
        const queued = navigator.sendBeacon(ENDPOINT, blob);
        if (!queued) this.queue.unshift.apply(this.queue, batch);
        return;
      }

      try {
        fetch(ENDPOINT, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: body,
          keepalive: true,
          credentials: 'same-origin',
        });
      } catch (error) {
        this.queue.unshift.apply(this.queue, batch);
      }
    }

    /** Start the periodic flush loop. */
    startFlushLoop() {
      this.timer = window.setInterval(() => this.flush(), FLUSH_INTERVAL_MS);
    }

    // --------------------------------------------------------- dwell tracking

    /** Accumulate visible time and reset the visibility clock. */
    accumulateVisibleTime() {
      if (this.visibleSince !== null) {
        this.visibleMs += Date.now() - this.visibleSince;
        this.visibleSince = null;
      }
    }

    /**
     * Queue a `time_spent` event for this page, once.
     * Only genuinely visible time counts — a tab left open in the background
     * must not look like deep engagement.
     */
    reportDwell() {
      if (this.dwellReported) return;
      this.accumulateVisibleTime();

      const seconds = Math.round(this.visibleMs / 1000);
      if (seconds < MIN_DWELL_SECONDS) return;

      this.dwellReported = true;
      const metadata = { seconds: seconds, path: this.path };
      if (this.productTitle) metadata.product_title = this.productTitle;

      this.track('time_spent', {
        product_id: this.productId,
        metadata: metadata,
      });
    }

    // -------------------------------------------------------------- listeners

    /** Flush pending events (including dwell) when the page goes away. */
    attachUnloadHandlers() {
      const finalise = () => {
        this.reportDwell();
        this.flushBeacon();
      };

      // `pagehide` and `visibilitychange` are far more reliable than
      // `beforeunload`, especially on iOS Safari. All three are attached; the
      // `dwellReported` guard makes duplicate firing harmless.
      window.addEventListener('pagehide', finalise);
      window.addEventListener('beforeunload', finalise);
      document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'hidden') finalise();
      });
    }

    /** Keep the dwell clock honest across tab switches. */
    attachVisibilityHandler() {
      document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'visible') {
          if (this.visibleSince === null) this.visibleSince = Date.now();
        } else {
          this.accumulateVisibleTime();
        }
      });
    }

    /**
     * One delegated listener for every product link on the page.
     *
     * Templates opt in declaratively with `data-track-click` (product id),
     * `data-track-source` (catalog / search / recommendation / …) and, for
     * recommended items, `data-track-recommendation`. Delegation means links
     * rendered later by the poller are tracked without re-binding anything.
     */
    attachClickDelegation() {
      document.addEventListener(
        'click',
        (event) => {
          const target = event.target instanceof Element
            ? event.target.closest('[data-track-click]')
            : null;
          if (!target) return;

          const productId = parseInt(target.getAttribute('data-track-click'), 10);
          if (Number.isNaN(productId)) return;

          const source = target.getAttribute('data-track-source') || 'unknown';
          const recommendationId = target.getAttribute('data-track-recommendation');

          this.track('product_click', {
            product_id: productId,
            metadata: { source: source },
          });

          // A click on a recommended item is the conversion signal that tells
          // the agent its last recommendation actually landed.
          if (recommendationId) {
            this.track('recommendation_click', {
              product_id: productId,
              metadata: {
                recommendation_id: parseInt(recommendationId, 10),
                source: source,
              },
            });
          }

          // Navigation is imminent, so get the batch out via beacon.
          this.flushBeacon();
        },
        true
      );
    }

    // ------------------------------------------------------- initial events

    /** Emit the `page_view` event for this page load. */
    trackInitialPageView() {
      const metadata = { referrer: document.referrer || null };
      if (this.productTitle) metadata.product_title = this.productTitle;

      this.track('page_view', {
        product_id: this.productId,
        metadata: metadata,
      });
    }

    /**
     * Emit a `search_query` event when the page was rendered for a search.
     * The result count comes from the server-rendered meta tag, so it is the
     * real number rather than a client-side guess.
     */
    trackSearchIfPresent() {
      const query = this.meta('smartreco-search-query');
      if (!query) return;

      const resultCount = this.metaInt('smartreco-search-results');
      this.track('search_query', {
        metadata: { query: query, result_count: resultCount === null ? 0 : resultCount },
      });
    }
  }

  // Expose a single instance so templates can emit bespoke events
  // (e.g. the add-to-cart button on the product page).
  window.smartRecoTracker = new SmartRecoTracker();
})();
