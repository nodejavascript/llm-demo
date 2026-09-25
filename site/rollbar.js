/* Rollbar error reporting for a record site.
 *
 * WHY THIS IS A FILE OF OURS AND NOT <script src="https://cdn.rollbar.com/...">.
 * The house ships no third-party JavaScript: the Google tag only loads after a yes, and a
 * site carrying tens of thousands of public records should not let a stranger's CDN run
 * code on every page. So Rollbar's own snippet is not used — this posts to the same relay
 * the snippet's host would need, and nothing else about the site changes.
 *
 * WRITTEN IN STANDARD STYLE — SINGLE QUOTES, NO SEMICOLONS — ON PURPOSE. The four games on
 * this domain lint with `standard`, and they lint this file because it lives in their
 * `public/` folder. The first version carried semicolons, so `npm run lint` failed and four
 * deploys stopped before they built. ONE reporter that the strictest linter in the fleet
 * accepts beats four copies that each need a lint exemption.
 *
 * THERE IS NO CREDENTIAL IN THIS PAGE, AND THAT IS MEASURED, NOT A PREFERENCE
 * (25 September 2026). Rollbar REFUSES a browser token on its REST item endpoint: 403
 * "insufficient privileges: post_server_item scope is required but the access token only
 * has post_client_item" — in every shape tried, including a request shaped exactly like a
 * browser's with Origin set. The token that CAN write an item is the server token, which
 * must never be in a page. So this file does not talk to Rollbar at all: it posts to
 * /api/fault ON ITS OWN ORIGIN, and the `record-fault` Worker relays it with the project's
 * server token, held in a Worker secret. A host the relay has no project for is accepted
 * and dropped, so a site without monitoring is still a normal site.
 *
 * WHAT IS SENT, AND WHAT IS DELIBERATELY NOT.
 *   Sent: the message, the source file, the line and column, the stack, the page PATH
 *   (without the query string or the fragment), and the browser's own user agent.
 *   NOT sent: the query string — a search term belongs to the reader, not to us — no
 *   cookie, no identifier, no session replay, no person record. Nothing at all is sent when
 *   the browser has asked not to be tracked.
 */
(function () {
  'use strict'

  // An error report is still a report about a person's visit. If they have asked not to be
  // tracked, the answer is no.
  if (navigator.doNotTrack === '1' || window.doNotTrack === '1') return

  const ENDPOINT = '/api/fault'
  const MAX_PER_PAGE = 5 // a loop that reports itself would be worse than the bug
  let sent = 0

  function post (body) {
    try {
      fetch(ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: body,
        keepalive: true,
        mode: 'cors',
        credentials: 'omit'
      }).catch(function () {
        // The relay being unreachable is not the reader's problem and not a second error.
      })
    } catch (e) {
      try {
        // The Blob is typed, so sendBeacon keeps Content-Type: application/json.
        navigator.sendBeacon(ENDPOINT, new Blob([body], { type: 'application/json' }))
      } catch (e2) {
        // A reporter must never throw into the page it is watching.
      }
    }
  }

  function report (message, source, lineno, colno, stack) {
    if (sent >= MAX_PER_PAGE) return
    sent += 1
    // PATH ONLY, and it is dropped HERE rather than at the relay, so the query never leaves
    // the browser at all. The relay composes the Rollbar item; this only describes the fault.
    post(JSON.stringify({
      host: location.host,
      path: location.pathname,
      message: String(message || 'error').slice(0, 400),
      source: source || null,
      line: lineno || null,
      column: colno || null,
      stack: stack ? String(stack).slice(0, 2000) : null,
      level: 'error',
      browser: navigator.userAgent
    }))
  }

  window.addEventListener('error', function (e) {
    report(e.message, e.filename, e.lineno, e.colno, e.error && e.error.stack)
  })

  window.addEventListener('unhandledrejection', function (e) {
    const r = e.reason
    report(
      'Unhandled promise rejection: ' + ((r && r.message) || r),
      location.pathname,
      null,
      null,
      r && r.stack
    )
  })
})()
