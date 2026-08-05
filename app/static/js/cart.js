/**
 * Nexora cart — client-side commerce state.
 *
 * Cart contents live in localStorage as `[{id, qty}]`. Product details (title,
 * price, cover) are resolved at render time from the catalog index the server
 * embeds in `#catalog-data`, so a saved cart can never show a stale price.
 *
 * Exposes `window.nexoraCart` for other modules and inline handlers.
 *
 * NOTE: no payment logic lives here or anywhere else in the app. Checkout is a
 * validated UI shell that renders a confirmation — nothing is charged or sent.
 *
 * @module cart
 */
(function () {
  'use strict';

  const KEY = 'nexora_cart_v1';

  /** @return {Array<{id:number, qty:number}>} */
  function read() {
    try {
      const raw = JSON.parse(localStorage.getItem(KEY) || '[]');
      return Array.isArray(raw)
        ? raw.filter(function (i) { return i && Number.isFinite(i.id); })
        : [];
    } catch (error) {
      return [];
    }
  }

  /** @param {Array} items */
  function write(items) {
    try {
      localStorage.setItem(KEY, JSON.stringify(items));
    } catch (error) {
      /* quota or private mode — the UI still works for this page view */
    }
    broadcast();
  }

  /** Catalog index embedded by the server (id → product). */
  function catalog() {
    const node = document.getElementById('catalog-data');
    if (!node) return {};
    try {
      const list = JSON.parse(node.textContent || '[]');
      const index = {};
      list.forEach(function (p) { index[p.id] = p; });
      return index;
    } catch (error) {
      return {};
    }
  }

  /** @return {number} total units in the cart */
  function count() {
    return read().reduce(function (sum, i) { return sum + (i.qty || 1); }, 0);
  }

  /** Update every cart badge and notify listeners. */
  function broadcast() {
    const n = count();
    document.querySelectorAll('[data-cart-count]').forEach(function (badge) {
      badge.textContent = String(n);
      badge.classList.toggle('hidden', n === 0);
      if (n > 0) {
        badge.classList.remove('saved-pop');
        void badge.offsetWidth;
        badge.classList.add('saved-pop');
      }
    });
    document.dispatchEvent(new CustomEvent('nexora:cart-changed', { detail: { count: n } }));
  }

  /**
   * Add a product to the cart.
   * @param {number} id
   * @param {number=} qty
   */
  function add(id, qty) {
    const productId = parseInt(id, 10);
    if (Number.isNaN(productId)) return;
    const items = read();
    const existing = items.find(function (i) { return i.id === productId; });
    if (existing) {
      existing.qty = Math.min(99, (existing.qty || 1) + (qty || 1));
    } else {
      items.push({ id: productId, qty: qty || 1 });
    }
    write(items);

    // Commerce intent is also behavioural signal — keep the agent in the loop.
    if (window.smartRecoTracker) {
      window.smartRecoTracker.track('add_to_cart', {
        product_id: productId,
        metadata: { source: 'cart_button' },
      });
    }
  }

  /** @param {number} id */
  function remove(id) {
    write(read().filter(function (i) { return i.id !== parseInt(id, 10); }));
  }

  /**
   * Set an explicit quantity (0 removes).
   * @param {number} id
   * @param {number} qty
   */
  function setQty(id, qty) {
    const productId = parseInt(id, 10);
    const next = Math.max(0, Math.min(99, parseInt(qty, 10) || 0));
    if (next === 0) return remove(productId);
    const items = read();
    const existing = items.find(function (i) { return i.id === productId; });
    if (existing) existing.qty = next;
    write(items);
  }

  /** Empty the cart. */
  function clear() {
    write([]);
  }

  /**
   * Resolve cart lines against the catalog index.
   * @return {{lines: Array<Object>, subtotal: number, tax: number, total: number, units: number}}
   */
  function summary() {
    const index = catalog();
    const lines = read()
      .map(function (item) {
        const product = index[item.id];
        if (!product) return null;
        const qty = item.qty || 1;
        return {
          id: product.id,
          title: product.title,
          price: product.price || 0,
          qty: qty,
          lineTotal: (product.price || 0) * qty,
          category: product.category,
          instructor: product.instructor,
          thumbnail_url: product.thumbnail_url,
          skill_level: product.skill_level,
          duration: product.duration,
        };
      })
      .filter(Boolean);

    const subtotal = lines.reduce(function (s, l) { return s + l.lineTotal; }, 0);
    // Illustrative VAT line so the totals block looks like a real checkout.
    const tax = Math.round(subtotal * 0.2 * 100) / 100;
    return {
      lines: lines,
      subtotal: subtotal,
      tax: tax,
      total: Math.round((subtotal + tax) * 100) / 100,
      units: lines.reduce(function (s, l) { return s + l.qty; }, 0),
    };
  }

  /**
   * Format a number as USD.
   * @param {number} value
   * @return {string}
   */
  function money(value) {
    const n = parseFloat(value);
    if (!n || Number.isNaN(n)) return 'Free';
    return '$' + n.toLocaleString('en-US', {
      minimumFractionDigits: 2, maximumFractionDigits: 2,
    });
  }

  window.nexoraCart = {
    read: read, add: add, remove: remove, setQty: setQty,
    clear: clear, count: count, summary: summary, money: money,
  };

  // ---- Delegated "Add to cart" / "Buy now" buttons -------------------------
  document.addEventListener('click', function (event) {
    const target = event.target instanceof Element ? event.target : null;
    if (!target) return;

    const addBtn = target.closest('[data-cart-add]');
    if (addBtn) {
      event.preventDefault();
      const id = addBtn.dataset.cartAdd;
      add(id, 1);

      addBtn.classList.remove('save-burst');
      void addBtn.offsetWidth;
      addBtn.classList.add('save-burst');

      const label = addBtn.querySelector('[data-cart-label]');
      if (label) {
        const original = addBtn.dataset.cartOriginal || label.textContent;
        addBtn.dataset.cartOriginal = original;
        label.textContent = 'Added ✓';
        addBtn.classList.add('is-added');
        window.setTimeout(function () {
          label.textContent = original;
          addBtn.classList.remove('is-added');
        }, 1600);
      }
      return;
    }

    const buyBtn = target.closest('[data-cart-buy]');
    if (buyBtn) {
      event.preventDefault();
      add(buyBtn.dataset.cartBuy, 1);
      window.location.href = '/checkout';
    }
  });

  broadcast();
})();
