/**
 * Saved-items ("Save for later") behaviour.
 *
 * Saves are recorded server-side as `add_to_cart` events, which is the same
 * signal the recommendation agent already weights most heavily — so saving a
 * course visibly sharpens the next recommendation instead of being an inert
 * bookmark. This module keeps the nav counter and every save button in sync.
 *
 * Buttons opt in declaratively:
 *   <button data-save-product="12" data-save-state="saved|unsaved">
 *
 * @module saved
 */
(function () {
  'use strict';

  const counter = document.getElementById('saved-count');

  /**
   * Update the nav badge, hiding it at zero.
   * @param {number} count
   */
  function setCount(count) {
    if (!counter) return;
    const n = parseInt(count, 10) || 0;
    counter.textContent = String(n);
    counter.classList.toggle('hidden', n === 0);
    if (n > 0) {
      counter.classList.remove('saved-pop');
      void counter.offsetWidth; // restart the animation
      counter.classList.add('saved-pop');
    }
  }

  /**
   * Paint a button for its current state.
   * @param {Element} button
   * @param {boolean} saved
   */
  function paint(button, saved) {
    button.dataset.saveState = saved ? 'saved' : 'unsaved';
    const label = button.querySelector('[data-save-label]');
    if (label) label.textContent = saved ? 'Saved' : (button.dataset.saveLabel || 'Save for later');
    button.setAttribute('aria-pressed', saved ? 'true' : 'false');
    button.classList.toggle('is-saved', saved);
  }

  /**
   * Toggle a save via the API.
   * @param {Element} button
   */
  async function toggle(button) {
    const productId = parseInt(button.dataset.saveProduct, 10);
    if (Number.isNaN(productId) || button.disabled) return;

    const currentlySaved = button.dataset.saveState === 'saved';
    button.disabled = true;

    try {
      const res = await fetch('/api/assistant/saved/' + productId, {
        method: currentlySaved ? 'DELETE' : 'POST',
        headers: { 'X-Requested-With': 'XMLHttpRequest' },
        credentials: 'same-origin',
      });
      if (!res.ok) throw new Error('save failed');

      const data = await res.json();
      paint(button, data.saved);
      setCount(data.count);

      if (data.saved) {
        button.classList.remove('save-burst');
        void button.offsetWidth;
        button.classList.add('save-burst');
        // Saving is a strong intent signal — tell the tracker too, so the
        // agent's trigger policy sees it immediately.
        if (window.smartRecoTracker) {
          window.smartRecoTracker.track('add_to_cart', {
            product_id: productId,
            metadata: { product_title: button.dataset.saveTitle || '', source: 'save_button' },
          });
        }
      }
    } catch (error) {
      // Restore the visible state; the user can retry.
      paint(button, currentlySaved);
    } finally {
      button.disabled = false;
    }
  }

  /** Sync the counter and every visible button with server state on load. */
  async function hydrate() {
    try {
      const res = await fetch('/api/assistant/saved', {
        headers: { 'X-Requested-With': 'XMLHttpRequest' },
        credentials: 'same-origin',
      });
      if (!res.ok) return;
      const data = await res.json();
      setCount(data.count);

      const savedIds = new Set((data.items || []).map(function (i) { return i.id; }));
      document.querySelectorAll('[data-save-product]').forEach(function (button) {
        paint(button, savedIds.has(parseInt(button.dataset.saveProduct, 10)));
      });
    } catch (error) {
      // Counter simply stays hidden.
    }
  }

  // Delegated so buttons rendered later (recommendation cards) work too.
  document.addEventListener('click', function (event) {
    const button = event.target instanceof Element
      ? event.target.closest('[data-save-product]')
      : null;
    if (!button) return;
    event.preventDefault();
    event.stopPropagation();
    toggle(button);
  });

  hydrate();
})();
