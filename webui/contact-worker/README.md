# Drowse contact delivery

Separate Cloudflare Worker for `POST https://drowse.ai/api/contact`. The Pages
site remains static; the Python server does not need Cloudflare credentials.
The shared feedback/contact form calls this endpoint without cookies.

## Activation checklist

Sending is deliberately disabled by `CONTACT_ENABLED = "false"`. A local test,
dry run, or successful website build does **not** establish live email delivery.

1. Authenticate Wrangler to the account that owns the `drowse.ai` zone, with
   permission to deploy Workers, manage the route and Durable Object, and inspect
   Email Service. Run `npx wrangler whoami` to confirm the intended account.
2. Follow Cloudflare's [sending-domain setup](https://developers.cloudflare.com/email-service/get-started/send-emails/).
   Inspect existing mail DNS first. Do not replace MX records or disrupt the
   current inbox provider to add an outbound contact form. If the apex cannot be
   configured safely, use an approved sending subdomain and update both the
   sender in `src/index.ts` and the binding's sender restriction.
3. Verify the sending domain and authorization to send from `feedback@drowse.ai`.
   Confirm that `contact@drowse.ai` is a working recipient, including any
   destination verification required by the account's Email Service setup.
   Check account plan, limits, and potential charges before activating services.
4. Confirm rate-limit namespace IDs `24001` and `24002` are not already used by
   another application in the account. Change them here if necessary.
5. Run the local checks below, then deploy the Worker with sending still disabled.
   Confirm `/api/contact` returns JSON `503` for an allowed-origin POST rather
   than the Pages HTML fallback. The route affects only `/api/contact*`.
6. Set `CONTACT_ENABLED` to `"true"` in `wrangler.jsonc`, regenerate types, and
   redeploy. Keep this value in source control so a later deploy does not revert
   dashboard-only configuration.
7. Submit clearly labeled synthetic messages through both public forms. Check
   that each matching reference actually reaches `contact@drowse.ai`, has all
   fields, and that Reply targets the optional address. Test the anonymous case
   too. Inspect inbox/spam and Cloudflare delivery events; a provider message ID
   proves acceptance, not arrival in the mailbox.

To disable sending, deploy with `CONTACT_ENABLED = "false"`. This leaves the
forms' draft recovery and direct-email links usable. Do not remove the site's
existing Pages routes, move its hosting, or alter inbox routing for this Worker.

## Local checks

From this directory:

```sh
npm ci
npm run types
npm run check
npm test
npm run build
```

Tests run in workerd and mock the email binding; they do not send real mail.
`npm run build` is a deployment dry run. `npm run deploy` performs a live deploy
and requires the account/setup checklist above. Do not use remote email bindings
for routine local development: [remote email bindings send real messages](https://developers.cloudflare.com/email-service/local-development/sending/).

From `webui`, run `npm run check` and
`npx playwright test e2e/contact.spec.ts`. Browser tests mock the endpoint and
exercise offline/error recovery, focus, mobile layouts, and duplicate submits.

## Delivery and privacy contract

- Sender and recipient are fixed in code and restricted in the binding. The
  visitor's optional email is only a Reply-To address, never the sender.
- Both plain-text and escaped HTML email include source form, reason, topic,
  optional reply email, title, body, received time, and submission reference.
  No chat text, model data, browser identifiers, or IP address is attached.
- Input is validated server-side, the streamed request is capped at 40 KB, and
  a honeypot rejects obvious bots. Only HTTPS public-site and HTTP/HTTPS loopback
  browser origins are allowed. Other native-server origins can use direct email.
- Limits are five attempts per minute per IP and 60 valid attempts per minute
  per Cloudflare location. These are best-effort abuse controls, not a global
  billing cap or complete bot protection. Add a verified challenge/WAF policy if
  real traffic requires it; do not rely on CORS to stop non-browser clients.
- Each random submission reference owns a Durable Object receipt. Retries reuse
  the reference. A claim is written before calling email; concurrent requests
  cannot send twice. Known provider rejections allow a retry. An ambiguous result
  remains pending and must be investigated by reference rather than resent
  automatically. Changing the form creates a new reference.
- Only a content hash, delivery status, and provider message ID are stored in the
  receipt. Active receipts expire after seven days; retry deduplication is not
  guaranteed beyond that window. Cloudflare backup/PITR retention may outlast
  active rows. The inbox naturally retains the email contents under its own
  retention policy.
- Application logs contain event names, references, provider IDs, and error
  codes, not form text or email addresses. The platform still processes ordinary
  request metadata. Frontend drafts exist only in memory while the page is open.
- Success requires JSON `status: "sent"` with the matching reference after Email
  Service returns a message ID. Offline, timeout, HTML fallback, rejection, and
  pending responses preserve the draft and never show success.

See [Workers email API](https://developers.cloudflare.com/email-service/api/send-emails/workers-api/)
and [binding restrictions](https://developers.cloudflare.com/email-service/configuration/send-bindings/).
