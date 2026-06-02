import { launch } from '/Volumes/Data/libs/npms/global/lib/node_modules/cloakbrowser/dist/index.js';
import { readFileSync, writeFileSync } from 'fs';

const USERNAME = '19245797470';
const PASSWORD = '99love';
const FILE_PATH = '/Volumes/Data/download/bendy-jizhang-v0.6.1-arm64-v8a.apk';
const FILE_NAME = 'bendy-jizhang-v0.6.1-arm64-v8a.apk';

async function main() {
  console.log('🚀 启动 CloakBrowser...');
  const browser = await launch({ headless: false });
  const context = await browser.newContext();
  const page = await context.newPage();

  // 拦截响应
  page.on('response', async res => {
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
      console.log('Body:', body);
    }
    if (url.includes('doupload.php') && postData) {
      const task = postData.match(/task=(\d+)/)?.[1];
      if (['2','3','5','6','22','47'].includes(task)) {
        console.log(`📡 doupload task=${task} → ${body.slice(0,200)}`);
      }
    }
    if (url.includes('accounts.woozooo.com/accounts.php') && req.method() === 'POST') {
      console.log('\n🔑 LOGIN RESPONSE:', body.slice(0, 300));
    }
  });

  // ── Step 1: 登录 ────────────────────────────────────────────────────
  console.log('\n=== 登录 ===');
  await page.goto('https://accounts.woozooo.com/accounts.php?action=login&ref=pc.woozooo.com', {
    waitUntil: 'networkidle',
  });
  await new Promise(r => setTimeout(r, 1000));

  // 填写表单
  await page.fill('#username', USERNAME);
  await page.fill('#password', PASSWORD);
  await page.screenshot({ path: 'step1_filled.png' });
  await page.click('#s3');

  await page.waitForNavigation({ waitUntil: 'networkidle', timeout: 20000 }).catch(() => {});
  await new Promise(r => setTimeout(r, 2000));
  console.log('登录后 URL:', page.url());
  await page.screenshot({ path: 'step2_after_login.png' });

  // ── Step 2: 确认登录态 + 抓 cookie ──────────────────────────────────
  const cookies = await context.cookies();
  const cookieMap = Object.fromEntries(cookies.map(c => [c.name, c.value]));
  const ylogin = cookieMap['ylogin'];
  console.log('\n🍪 ylogin:', ylogin);
  console.log('全部 cookies:', JSON.stringify(cookieMap, null, 2));

  if (!ylogin || ylogin === '0') {
    console.log('❌ 登录失败，退出');
    await browser.close();
    return;
  }

  // 保存 cookie 供 Python 使用
  writeFileSync('cookies.json', JSON.stringify(cookieMap, null, 2));
  console.log('✅ Cookie 已保存到 cookies.json');

  // ── Step 3: 访问 mydisk ──────────────────────────────────────────────
  await page.goto('https://pc.woozooo.com/mydisk.php', { waitUntil: 'networkidle' });
  await new Promise(r => setTimeout(r, 3000));
  await page.screenshot({ path: 'step3_mydisk.png' });

  // 读取文件内容
  console.log(`\n📂 读取文件: ${FILE_PATH}`);
  const fileContent = readFileSync(FILE_PATH);
  console.log(`文件大小: ${(fileContent.length / 1024 / 1024).toFixed(2)} MB`);
  const base64Content = fileContent.toString('base64');

  // ── Step 4: 在页面内用 fetch 上传 ────────────────────────────────────
  console.log('\n🔄 开始上传...');
  const uploadResult = await page.evaluate(async ({ b64, fileName }) => {
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    const blob = new Blob([bytes], { type: 'application/octet-stream' });

    const form = new FormData();
    form.append('task', '1');
    form.append('vie', '2');
    form.append('ve', '2');
    form.append('id', 'WU_FILE_0');
    form.append('name', fileName);
    form.append('folder_id_bb_n', '-1');
    form.append('upload_file', blob, fileName);

    const resp = await fetch('https://pc.woozooo.com/fileup.php', {
      method: 'POST',
      body: form,
      credentials: 'include',
    });
    const text = await resp.text();
    return { ok: resp.ok, status: resp.status, body: text };
  }, { b64: base64Content, fileName: FILE_NAME });

  console.log('\n📦 上传结果:', JSON.stringify(uploadResult, null, 2));

  await new Promise(r => setTimeout(r, 5000));
  await browser.close();
}

main().catch(console.error);
