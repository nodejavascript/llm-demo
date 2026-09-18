"use strict";
/**
 * consent.ts — the cookie gate, and the only thing on this site that may load
 * Google's script.
 *
 * The model is the same one inputresponse.com uses, and it is the same for the
 * same reason. The law that shapes it is the European one, broadly copied now by
 * the United Kingdom, Brazil, Quebec and others: storage that is not strictly
 * necessary for the thing the visitor actually asked for may not be set until
 * they say yes, and saying no has to be as easy as saying yes. Counting somebody
 * as a statistic because they loaded a page is the exact thing the rule was
 * written about.
 *
 * So nothing here is clever:
 *
 *   - **The Google Analytics script is NOT in the page.** It is appended only
 *     after a yes, which means a declined visit makes no request to Google at all
 *     and receives no cookie. Not "cookies disabled" — never loaded. End-to-end
 *     tests assert both halves: no consent, no request, `window.gtag` undefined.
 *   - The choice is remembered in localStorage rather than a cookie, because
 *     setting a cookie to record a cookie decision would be its own small joke.
 *   - Accept and Reject are the same size, the same weight and the same distance
 *     from the reader. There is no pre-ticked box and no "manage preferences"
 *     maze with the off switch three screens down.
 *   - Rejecting is a real answer, not a nag: the bar does not return on the next
 *     visit.
 *   - The longer answer is a panel, not a maze. Every purpose is named rather
 *     than numbered, and the ONE switch starts off — a box that arrives already
 *     ticked is a default, not a choice.
 *
 * WHAT IS DIFFERENT HERE, and it is a difference in the visitor's favour: this
 * site has **no strictly necessary cookie at all**. inputresponse needs one to
 * carry a player's record between visits; llm-demo has no accounts and stores
 * nothing about the visitor — the model a visitor trains lives in that tab's
 * memory and is gone when the tab closes. So this panel names ONE real choice
 * instead of inventing categories to look thorough, and the ask says plainly that
 * the text you paste is never sent anywhere whatever you answer here. That
 * sentence matters more on this page than on any other site in the family: the
 * whole demo is asking people to paste something they care about.
 */
(function () {
    var KEY = 'analytics_consent';
    var OWNER_KEY = 'ga_opt_out';
    var started = false;
    function read(key) {
        try {
            return localStorage.getItem(key);
        }
        catch (error) {
            return null;
        }
    }
    function write(key, value) {
        try {
            localStorage.setItem(key, value);
        }
        catch (error) {
            /* Private mode, or a browser with storage switched off. The choice then
               lasts for the visit, which is the honest outcome. */
        }
    }
    function drop(key) {
        try {
            localStorage.removeItem(key);
        }
        catch (error) {
            /* as above */
        }
    }
    function measurementId() {
        var tag = document.querySelector('script[data-ga-id]');
        return tag ? tag.getAttribute('data-ga-id') || '' : '';
    }
    function ownerOptedOut() {
        return read(OWNER_KEY) === '1';
    }
    function allowed() {
        return !ownerOptedOut() && read(KEY) === 'granted';
    }
    /**
     * Load the analytics tag, once, and only if it is allowed. Nothing before this
     * point has contacted Google.
     */
    function start() {
        if (started || !allowed())
            return;
        var id = measurementId();
        if (!id)
            return;
        started = true;
        window.dataLayer = window.dataLayer || [];
        window.gtag = function gtag() {
            // eslint-disable-next-line prefer-rest-params
            window.dataLayer.push(arguments);
        };
        // The page's own events go out through this, and it only exists once the
        // visitor has said yes — a page with no consent has no way to send one.
        window.llmTrack = function llmTrack(name, params = {}) {
            window.gtag?.('event', name, params);
        };
        var tag = document.createElement('script');
        tag.async = true;
        tag.src = 'https://www.googletagmanager.com/gtag/js?id=' + encodeURIComponent(id);
        document.head.appendChild(tag);
        window.gtag('js', new Date());
        // The page sends its own page_view, so the automatic one stays off rather
        // than counting this landing twice.
        window.gtag('config', id, { send_page_view: false });
        window.gtag('event', 'page_view', {
            page_location: location.href,
            page_path: location.pathname,
            page_title: document.title,
        });
        // The other two things the house standard asks every site to send, both of
        // them about the page rather than about the reader: which controls are
        // pressed, and how far down a page people get.
        //
        // Two deliberate limits. **Nothing here reads what anybody typed** — a
        // control is named by its own id, its tag and its role, never by its
        // contents, and the training text is never read at all. And **the listeners
        // do not exist until this point**: a visitor who has not allowed analytics
        // has no click or scroll listener on the page at all, rather than one that
        // stays quiet, which is the difference between a promise and a habit.
        document.addEventListener('click', function (event) {
            var target = event.target;
            var node = target && target.closest ? target.closest('a, button') : null;
            if (!node)
                return;
            var href = node.getAttribute('href') || '';
            var offsite = /^https?:\/\//.test(href) && href.indexOf(location.host) === -1;
            var scheme = href.indexOf('mailto:') === 0 ? 'mailto' : href.indexOf('tel:') === 0 ? 'tel' : '';
            window.gtag?.('event', 'element_click', {
                page_path: location.pathname,
                element_id: node.id || '',
                element_kind: node.tagName.toLowerCase(),
                element_role: node.getAttribute('data-ga') || String(node.className || '').split(' ')[0] || '',
                outbound: offsite || scheme === 'mailto',
                link_scheme: scheme,
                outbound_host: offsite ? new URL(href).host : '',
            });
        }, true);
        var marks = [25, 50, 75, 100];
        var deepest = 0;
        var sent = {};
        var onScroll = function () {
            var page = document.documentElement.scrollHeight;
            // A page shorter than the window is fully read the moment it is opened, so
            // it counts as the deepest mark rather than as nothing at all.
            var percent = page <= window.innerHeight
                ? 100
                : Math.min(100, Math.round(((window.scrollY + window.innerHeight) / page) * 100));
            deepest = Math.max(deepest, percent);
            for (var i = 0; i < marks.length; i += 1) {
                var mark = marks[i];
                if (deepest >= mark && !sent[mark]) {
                    sent[mark] = true;
                    window.gtag?.('event', 'scroll_depth', { page_path: location.pathname, percent_scrolled: mark });
                }
            }
        };
        window.addEventListener('scroll', onScroll, { passive: true });
        onScroll();
    }
    /**
     * The owner's own switch, from the address bar: ?ga=off / ?ga=on. It beats
     * consent, because it exists so the person who runs the site can leave himself
     * out of his own numbers without changing what every visitor is offered.
     */
    function applyOwnerSwitch() {
        var params = new URLSearchParams(location.search);
        var flag = params.get('ga');
        if (flag === 'off') {
            write(OWNER_KEY, '1');
            // Already loaded on this page? Reloading is what actually stops it.
            if (window.gtag)
                location.reload();
        }
        if (flag === 'on')
            drop(OWNER_KEY);
    }
    /**
     * The footer's counting switch, worded for what it will actually do, and shown
     * only once it has been used.
     *
     * The row is the OWNER's control, not a visitor's. A visitor already has the
     * switch above it, so a second way to opt out of counting sitting under the
     * first is the same answer twice, in different words. It stays out of sight
     * until it has actually been used — and then it is the one click back, which is
     * the whole reason it is a row at all: turning counting off from the address
     * bar and having to retype the address to undo it was a dead end.
     */
    function wordCountSwitch() {
        var off = ownerOptedOut();
        var row = document.getElementById('consentDeviceRow');
        if (row)
            row.hidden = !off;
        var link = document.getElementById('countToggle');
        if (!link)
            return;
        link.textContent = off ? 'Count my visits' : 'Stop counting my visits';
        link.setAttribute('href', off ? '/?ga=on' : '/?ga=off');
    }
    var bar = null;
    /**
     * Reserve the banner's own height at the bottom of the page.
     *
     * The banner is fixed to the bottom of the window, which means it sits on top
     * of whatever is behind it and takes the click — on inputresponse it cost a
     * real visitor a real click on the footer, which was unclickable until they
     * answered. The page is told to leave exactly the banner's own height free,
     * measured here rather than guessed.
     *
     * Measuring once is not enough: anything can re-wrap the text after the first
     * paint, so the reservation follows the banner instead of sampling it — an
     * observer on the banner, plus a load event for the last late layout, and a
     * guard so a resize that does not change the height does not write the same
     * value and bounce back through the observer.
     */
    function reserveSpace() {
        var root = document.documentElement;
        var height = !bar || bar.hidden ? 0 : Math.ceil(bar.getBoundingClientRect().height);
        if (root.style.getPropertyValue('--consent-height') === height + 'px')
            return;
        root.style.setProperty('--consent-height', height + 'px');
    }
    function hide() {
        if (bar)
            bar.hidden = true;
        reserveSpace();
    }
    /**
     * Two views share this one fixed bar rather than a second dialog appearing
     * over it. That is not tidiness: the bar is the thing whose height is measured
     * and reserved at the bottom of the page, and the observer already follows it,
     * so the taller view is measured rather than guessed at.
     */
    function panel() {
        return {
            ask: document.getElementById('consentAsk'),
            prefs: document.getElementById('consentPrefs'),
            settings: document.getElementById('consentSettings'),
            toggle: document.getElementById('consentAnalytics'),
            word: document.getElementById('consentAnalyticsWord'),
        };
    }
    /**
     * The switch, and the word beside it is not decoration: a state carried only by
     * colour is a state some readers cannot see.
     *
     * The visual state is drawn from `aria-checked` in the stylesheet, so the ring
     * around the switch and what a screen reader announces cannot disagree — they
     * are the same attribute. Nothing here toggles a class.
     */
    function setAnalytics(on) {
        var parts = panel();
        if (parts.toggle)
            parts.toggle.setAttribute('aria-checked', on ? 'true' : 'false');
        if (parts.word)
            parts.word.textContent = on ? 'On' : 'Off';
    }
    function analyticsOn() {
        var parts = panel();
        return !!parts.toggle && parts.toggle.getAttribute('aria-checked') === 'true';
    }
    /**
     * Show the panel, with the switch reflecting whatever is already decided. It is
     * off when nothing has been decided, which is the same thing the reader was
     * offered the first time.
     */
    function openPrefs() {
        if (!bar)
            return;
        var parts = panel();
        setAnalytics(read(KEY) === 'granted');
        if (parts.ask)
            parts.ask.hidden = true;
        if (parts.prefs)
            parts.prefs.hidden = false;
        if (parts.settings)
            parts.settings.hidden = true;
        bar.hidden = false;
        reserveSpace();
        // Focus the panel, not a control inside it. A focus ring on a switch is a
        // recommendation about what to do with the switch, and this is the one place
        // on the site where a recommendation is the whole thing being avoided.
        if (parts.prefs) {
            parts.prefs.setAttribute('tabindex', '-1');
            parts.prefs.focus();
        }
    }
    /** Back to the question, which is the state the bar opens in. */
    function askMode() {
        var parts = panel();
        if (parts.ask)
            parts.ask.hidden = false;
        if (parts.prefs)
            parts.prefs.hidden = true;
        if (parts.settings)
            parts.settings.hidden = false;
    }
    function show() {
        if (!bar)
            return;
        askMode();
        bar.hidden = false;
        reserveSpace();
        // Focus the banner, NOT the first button. Putting focus on a button draws the
        // browser's focus ring around it, which quietly recommends that answer — and
        // whichever one it lands on, a recommendation is not what this is. The
        // container takes focus so a screen reader reaches the question immediately
        // and the reader still chooses with Tab.
        bar.setAttribute('tabindex', '-1');
        bar.focus();
    }
    function decide(answer) {
        var was = read(KEY);
        write(KEY, answer);
        hide();
        if (answer === 'granted')
            start();
        // Withdrawing has to actually stop it. The tag is already in the page and
        // cannot be unloaded, so the honest way to stop counting is to load the page
        // again without it — the same thing the address-bar switch does, and the only
        // version of "no" that is true.
        if (answer === 'denied' && was === 'granted' && window.gtag)
            location.reload();
    }
    /**
     * What moving the switch does: the answer, and the bar closes.
     *
     * It does not wait for a separate Save button, which would make one setting
     * into a two-step form — and somebody who flipped the switch and walked away
     * would have answered nothing while appearing to have answered something. The
     * switch IS the answer, exactly as Accept all and Reject all are.
     */
    function setAndDecide(on) {
        setAnalytics(on);
        decide(on ? 'granted' : 'denied');
    }
    /**
     * The x at the top of the panel, and the Escape key, which do the same thing.
     *
     * What closing MEANS depends on whether there is already an answer, and the
     * difference matters:
     *
     *   - an answer exists: the bar goes away and nothing changes. Somebody who
     *     opened the panel to look has to be able to leave without re-answering — a
     *     panel with no way out is a wall, not a setting.
     *   - nothing is decided yet: the panel steps back to the QUESTION. An x that
     *     dismissed the question would be a way to never choose and still be
     *     counted, and there is nothing to leave at that point anyway.
     */
    function closePrefs() {
        if (!read(KEY)) {
            show();
            return;
        }
        hide();
    }
    function build() {
        if (!measurementId())
            return;
        bar = document.getElementById('consentBar');
        if (!bar)
            return;
        var accept = document.getElementById('consentAccept');
        var decline = document.getElementById('consentDecline');
        if (accept)
            accept.addEventListener('click', function () { decide('granted'); });
        if (decline)
            decline.addEventListener('click', function () { decide('denied'); });
        // The longer answer. Settings opens the panel; the switch inside it IS the
        // answer, and moving it writes and closes — see setAndDecide.
        var parts = panel();
        if (parts.settings)
            parts.settings.addEventListener('click', openPrefs);
        if (parts.toggle) {
            parts.toggle.addEventListener('click', function () {
                setAndDecide(!analyticsOn());
            });
        }
        // The x, and Escape, which a reader will try whether or not it is offered.
        var close = document.getElementById('consentClose');
        if (close)
            close.addEventListener('click', closePrefs);
        document.addEventListener('keydown', function (event) {
            var prefs = document.getElementById('consentPrefs');
            if (event.key === 'Escape' && prefs && !prefs.hidden) {
                event.preventDefault();
                closePrefs();
            }
        });
        window.addEventListener('resize', reserveSpace);
        window.addEventListener('orientationchange', reserveSpace);
        window.addEventListener('load', reserveSpace);
        // The one that actually settles it: the banner tells us when its own size
        // moves, whatever the reason.
        if (typeof ResizeObserver === 'function')
            new ResizeObserver(reserveSpace).observe(bar);
        // The footer door. It opens the PANEL rather than asking the question again:
        // the reader has already answered, and something called settings that repeats
        // the question is not a settings control. The answer is left alone, so nothing
        // stops running just because somebody looked.
        var reopen = document.getElementById('consentBtn');
        if (reopen)
            reopen.addEventListener('click', openPrefs);
        if (!read(KEY)) {
            // Nothing shows while the visitor decides nothing: analytics is not running,
            // so the banner is the only thing asking.
            show();
        }
    }
    applyOwnerSwitch();
    // After the switch is applied, so the words match the state the reader is about
    // to be in rather than the one they arrived with.
    wordCountSwitch();
    start();
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', build);
    }
    else {
        build();
    }
})();
