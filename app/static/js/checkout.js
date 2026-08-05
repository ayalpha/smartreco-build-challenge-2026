/**
 * Checkout — client-side validation and a simulated payment.
 *
 * IMPORTANT: this file processes nothing. There is no gateway, no network call
 * with card data, no persistence. `pay()` validates the form, shows a spinner,
 * clears the cart and navigates to the confirmation page. Card values never
 * leave the input elements.
 *
 * Validation implemented here: Luhn checksum on the card number, expiry parsing
 * with a real not-in-the-past check, CVV length keyed to card brand, and
 * required-field checks with inline messages.
 *
 * @module checkout
 */
(function () {
  'use strict';

  const cart = window.nexoraCart;
  const form = document.getElementById('payment-form');
  if (!cart || !form) return;

  const el = {
    empty: document.getElementById('checkout-empty'),
    body: document.getElementById('checkout-body'),
    lines: document.getElementById('ck-lines'),
    subtotal: document.getElementById('ck-subtotal'),
    tax: document.getElementById('ck-tax'),
    total: document.getElementById('ck-total'),
    payTotal: document.getElementById('ck-pay-total'),
    pay: document.getElementById('ck-pay'),
    payLabel: document.getElementById('ck-pay-label'),
    spinner: document.getElementById('ck-pay-spinner'),
    stripe: document.getElementById('ck-stripe'),
    card: document.getElementById('ck-card'),
    expiry: document.getElementById('ck-expiry'),
    cvv: document.getElementById('ck-cvv'),
    brand: document.getElementById('ck-brand'),
  };

  function esc(v) {
    return String(v === null || v === undefined ? '' : v)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  // ------------------------------------------------------------- summary

  function renderSummary() {
    const s = cart.summary();
    const has = s.lines.length > 0;
    el.empty.classList.toggle('hidden', has);
    el.body.classList.toggle('hidden', !has);
    if (!has) return;

    el.lines.innerHTML = s.lines.map(function (l) {
      return '<li class="flex items-start justify-between gap-3">' +
        '<span class="min-w-0">' +
          '<span class="block truncate text-[13px] font-medium text-mist-100">' + esc(l.title) + '</span>' +
          '<span class="font-mono text-[10px] lowercase text-mist-500">qty ' + l.qty + '</span>' +
        '</span>' +
        '<span class="shrink-0 font-mono text-[13px] tabular-nums text-mist-200">' +
          cart.money(l.lineTotal) + '</span>' +
      '</li>';
    }).join('');

    el.subtotal.textContent = cart.money(s.subtotal);
    el.tax.textContent = cart.money(s.tax);
    el.total.textContent = cart.money(s.total);
    el.payTotal.textContent = cart.money(s.total);
  }

  // ------------------------------------------------------------ formatting

  /** Detect brand from the leading digits. */
  function brandOf(digits) {
    if (/^4/.test(digits)) return { name: 'visa', cvv: 3, len: [16] };
    if (/^(5[1-5]|2[2-7])/.test(digits)) return { name: 'mastercard', cvv: 3, len: [16] };
    if (/^3[47]/.test(digits)) return { name: 'amex', cvv: 4, len: [15] };
    if (/^6/.test(digits)) return { name: 'discover', cvv: 3, len: [16] };
    return { name: '', cvv: 3, len: [15, 16, 19] };
  }

  el.card.addEventListener('input', function () {
    const digits = el.card.value.replace(/\D/g, '').slice(0, 19);
    const brand = brandOf(digits);
    const groups = brand.name === 'amex'
      ? [digits.slice(0, 4), digits.slice(4, 10), digits.slice(10, 15)]
      : digits.match(/.{1,4}/g) || [];
    el.card.value = groups.filter(Boolean).join(' ');
    el.brand.textContent = brand.name;
    el.cvv.maxLength = brand.cvv;
  });

  el.expiry.addEventListener('input', function () {
    const digits = el.expiry.value.replace(/\D/g, '').slice(0, 4);
    el.expiry.value = digits.length > 2 ? digits.slice(0, 2) + '/' + digits.slice(2) : digits;
  });

  el.cvv.addEventListener('input', function () {
    el.cvv.value = el.cvv.value.replace(/\D/g, '');
  });

  // ------------------------------------------------------------ validation

  /** Luhn checksum. */
  function luhn(digits) {
    let sum = 0;
    let alt = false;
    for (let i = digits.length - 1; i >= 0; i--) {
      let n = parseInt(digits[i], 10);
      if (alt) { n *= 2; if (n > 9) n -= 9; }
      sum += n;
      alt = !alt;
    }
    return digits.length > 0 && sum % 10 === 0;
  }

  /**
   * Find the `.ck-error` node belonging to a field.
   *
   * Most fields are `<div><label><input class="ck-field"><p class="ck-error">`,
   * so the error node is a child of the input's immediate parent. The card
   * number is not: its input lives inside a `.relative` wrapper (which
   * positions the brand badge), and `.ck-error` is a SIBLING of that wrapper.
   * A plain `closest('div')` therefore lands on a container holding no error
   * node, and the card's message silently never renders.
   *
   * So walk up instead — and only accept an ancestor holding exactly one
   * `.ck-error`, which keeps us from grabbing a neighbour's node out of a
   * shared row wrapper (e.g. the expiry/CVV grid holds two).
   */
  function errorNodeFor(input) {
    let holder = input.parentElement;
    for (let depth = 0; holder && holder !== form && depth < 4; depth++) {
      const found = holder.querySelectorAll('.ck-error');
      if (found.length === 1) return found[0];
      holder = holder.parentElement;
    }
    return null;
  }

  function setError(input, message) {
    const node = errorNodeFor(input);
    const bad = Boolean(message);
    input.classList.toggle('border-rose-500/60', bad);
    input.setAttribute('aria-invalid', bad ? 'true' : 'false');
    if (node) {
      node.textContent = message || '';
      node.classList.toggle('hidden', !bad);
    }
    return !bad;
  }

  function validateField(input) {
    const value = input.value.trim();

    if (input === el.card) {
      const digits = value.replace(/\D/g, '');
      const brand = brandOf(digits);
      if (!digits) return setError(input, 'Card number is required.');
      if (brand.len.indexOf(digits.length) === -1) {
        return setError(input, 'That card number looks the wrong length.');
      }
      if (!luhn(digits)) return setError(input, 'That card number is not valid.');
      return setError(input, '');
    }

    if (input === el.expiry) {
      const m = value.match(/^(\d{2})\/(\d{2})$/);
      if (!m) return setError(input, 'Use MM/YY.');
      const month = parseInt(m[1], 10);
      const year = 2000 + parseInt(m[2], 10);
      if (month < 1 || month > 12) return setError(input, 'Month must be 01–12.');
      const now = new Date();
      const end = new Date(year, month, 1); // first instant after the expiry month
      if (end <= now) return setError(input, 'That card has expired.');
      return setError(input, '');
    }

    if (input === el.cvv) {
      const need = brandOf(el.card.value.replace(/\D/g, '')).cvv;
      if (value.length !== need) return setError(input, need + ' digits required.');
      return setError(input, '');
    }

    if (input.type === 'email') {
      if (!value) return setError(input, 'Email is required.');
      if (!/^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(value)) {
        return setError(input, 'Enter a valid email address.');
      }
      return setError(input, '');
    }

    if (input.required && !value) {
      const label = form.querySelector('label[for="' + input.id + '"]');
      return setError(input, (label ? label.textContent.trim() : 'This field') + ' is required.');
    }
    return setError(input, '');
  }

  form.querySelectorAll('.ck-field').forEach(function (input) {
    input.addEventListener('blur', function () { validateField(input); });
    input.addEventListener('input', function () {
      if (input.getAttribute('aria-invalid') === 'true') validateField(input);
    });
  });

  function validateAll() {
    let firstBad = null;
    form.querySelectorAll('.ck-field').forEach(function (input) {
      if (!validateField(input) && !firstBad) firstBad = input;
    });
    if (firstBad) {
      firstBad.focus();
      firstBad.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
    return !firstBad;
  }

  // --------------------------------------------------------------- pay

  /**
   * Simulate payment: validate, show progress, clear cart, confirm.
   * No network request is made with any form value.
   */
  function pay() {
    if (!validateAll()) return;

    el.pay.disabled = true;
    el.stripe.disabled = true;
    el.spinner.classList.remove('hidden');
    el.payLabel.textContent = 'Processing…';

    window.setTimeout(function () {
      const s = cart.summary();
      try {
        sessionStorage.setItem('nexora_last_order', JSON.stringify({
          total: s.total,
          units: s.units,
          items: s.lines.map(function (l) {
            return { id: l.id, title: l.title, qty: l.qty, price: l.price };
          }),
          email: (document.getElementById('ck-email') || {}).value || '',
        }));
      } catch (error) { /* confirmation degrades to a generic message */ }

      cart.clear();
      window.location.href = '/checkout/success';
    }, 1400);
  }

  form.addEventListener('submit', function (event) {
    event.preventDefault();   // never posts anywhere
    pay();
  });

  el.stripe.addEventListener('click', function (event) {
    event.preventDefault();
    pay();
  });

  document.addEventListener('nexora:cart-changed', renderSummary);
  renderSummary();
})();
