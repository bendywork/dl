/**
 * 进入 mainframe iframe，找到真实上传控件，注入文件触发上传并拦截完整请求
 */
import { launch } from '/Volumes/Data/libs/npms/global/lib/node_modules/cloakbrowser/dist/index.js';
import { readFileSync, writeFileSync } from 'fs';

const FILE_PATH = '/Volumes/Data/download/bendy-jizhang-v0.6.1-arm64-v8a.apk';
const FILE_NAME = 'bendy-jizhang-v0.6.1-arm64-v8a.apk';
const USERNAME  = '19245797470';
const PASSWORD  = '99love';

async function main() {
  console.log('🚀 启动浏览器...');
  const browser = await launch({ headless: false });
  const context = await browser.newContext();
  const page = await context.newPage();

  // 拦截所有页面（含 iframe）的 fileup.php
  context.on('page', p => setupIntercept(p));
  setupIntercept(page);

  function setupIntercept(p) {
    p.on('response', async res => {
      const url = res.url();
      if (!url.includes('woozooo.com')) return;
      let body = '';
      try { body = await res.text(); } catch {}
      const req = res.request();
      let postData = '';
      try { postData = req.postData() || ''; } catch {}

      if (url.includes('fileup.php')) {
        console.log('\n📤 ===== UPLOAD RESPONSE =====');
        console.log('Status:', res.status());
        console.log('Body:', body || '(empty)');
        writeFileSync('upload_result.json', JSON.stringify({ status: res.status(), body }, null, 2));
      }
      if (url.includes('doupload.php') && postData) {
        const task = postData.match(/task=(\d+)/)?.[1];
        const vei  = postData.match(/vei=([^&\s]+)/)?.[1] || '';
        if (task) console.log(`📡 task=${task} vei=${vei} → ${body.slice(0,120)}`);
      }
    });

    // 拦截 fileup.php，打印完整请求头
    p.route('**/fileup.php', async route => {
      const req = route.request();
      const hdrs = req.headers();
      console.log('\n📤 REQUEST HEADERS:', JSON.stringify(hdrs, null, 2));
      const postData = req.postData() || '';
      console.log('POST DATA preview:', postData.slice(0, 200));
      const resp = await route.fetch();
      const body = await resp.text();
      console.log('RESPONSE:', body || '(empty)');
      writeFileSync('upload_capture.json', JSON.stringify({ headers: hdrs, status: resp.status(), body }, null, 2));
      await route.fulfill({ response: resp });
    });
  }

  // ── 登录 ─────────────────────────────────────────────────────────
  console.log('登录中...');
  await page.goto('https://accounts.woozooo.com/accounts.php?action=login&ref=pc.woozooo.com', { waitUntil: 'networkidle' });
  await page.fill('#username', USERNAME);
  await page.fill('#password', PASSWORD);
  await page.click('#s3');
  await page.waitForNavigation({ waitUntil: 'networkidle', timeout: 20000 }).catch(() => {});
  await new Promise(r => setTimeout(r, 2000));

  const cookies = await context.cookies();
  const ylogin = cookies.find(c => c.name === 'ylogin')?.value;
  console.log('✅ ylogin:', ylogin);
  if (!ylogin || ylogin === '0') { console.log('❌ 登录失败'); await browser.close(); return; }

  const cookieMap = Object.fromEntries(cookies.map(c => [c.name, c.value]));
  writeFileSync('cookies.json', JSON.stringify(cookieMap, null, 2));

  // ── 访问 mydisk ───────────────────────────────────────────────────
  await page.goto('https://pc.woozooo.com/mydisk.php', { waitUntil: 'networkidle' });
  await new Promise(r => setTimeout(r, 3000));

  // ── 进入 iframe ───────────────────────────────────────────────────
  const iframeEl = await page.$('#mainframe');
  if (!iframeEl) { console.log('❌ 找不到 #mainframe'); await page.screenshot({ path: 'debug.png' }); }

  const frame = await iframeEl?.contentFrame();
  console.log('iframe URL:', frame?.url());

  // 在 iframe 内搜索所有 input
  if (frame) {
    const inputs = await frame.$$eval('input', els => els.map(e => ({
      tag: e.tagName, type: e.type, id: e.id, name: e.name, cls: e.className
    })));
    console.log('iframe inputs:', JSON.stringify(inputs, null, 2));

    // 找 file input
    const fileInput = await frame.$('input[type="file"]');
    if (fileInput) {
      console.log('✅ 找到 file input，注入文件...');
      await fileInput.setInputFiles(FILE_PATH);
      console.log('⏳ 文件已注入，等待上传...');
      await new Promise(r => setTimeout(r, 60000));
    } else {
      console.log('❌ iframe 内没有 file input，截图并打印 HTML...');
      await frame.waitForLoadState('networkidle').catch(() => {});
      const html = await frame.content();
      writeFileSync('iframe_html.html', html);
      console.log('HTML 已保存到 iframe_html.html');
      await page.screenshot({ path: 'iframe_debug.png', fullPage: true });
      // 等用户手动操作
      console.log('⏳ 等待60秒，请手动操作浏览器上传文件...');
      await new Promise(r => setTimeout(r, 60000));
    }
  }

  await browser.close();
}

main().catch(console.error);
