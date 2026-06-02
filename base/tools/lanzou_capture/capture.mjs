import { launch } from '/Volumes/Data/libs/npms/global/lib/node_modules/cloakbrowser/dist/index.js';

const USERNAME = '19245797470';
const PASSWORD = '99love';

const captured = {
  login: null,
  folders: [],
  files: [],
  upload: null,
  createFolder: null,
  deleteFile: null,
  deleteFolder: null,
};

function logRequest(url, method, postData, response, responseBody) {
  const entry = { url, method, postData, status: response?.status(), responseBody };

  // 登录
  if (url.includes('account.php') && postData?.includes('task=3')) {
    captured.login = entry;
    console.log('\n✅ [LOGIN REQUEST]', url);
    console.log('   POST:', postData);
    console.log('   Response:', responseBody?.slice(0, 200));
  }

  // doupload.php 各种操作
  if (url.includes('doupload.php')) {
    const task = postData?.match(/task=(\d+)/)?.[1];
    console.log(`\n📡 [DOUPLOAD task=${task}]`, url);
    console.log('   POST:', postData);
    console.log('   Response:', responseBody?.slice(0, 300));

    if (task === '47' || task === '19') captured.folders.push(entry);
    if (task === '5') captured.files.push(entry);
    if (task === '2') captured.createFolder = entry;
    if (task === '3') captured.deleteFolder = entry;
    if (task === '6') captured.deleteFile = entry;
    if (task === '22') {
      console.log('🔗 [DIRECT LINK REQUEST]');
      captured.directLink = entry;
    }
  }

  // 上传
  if (url.includes('fileup.php')) {
    captured.upload = entry;
    console.log('\n📤 [UPLOAD REQUEST]', url);
    console.log('   POST keys:', postData?.slice(0, 200));
    console.log('   Response:', responseBody?.slice(0, 200));
  }

  // ajaxm.php 直链解析
  if (url.includes('ajaxm.php')) {
    console.log('\n🔗 [AJAXM - direct link resolve]', url);
    console.log('   POST:', postData);
    console.log('   Response:', responseBody?.slice(0, 300));
    captured.ajaxm = entry;
  }
}

async function main() {
  console.log('🚀 启动 CloakBrowser...');
  const browser = await launch({ headless: false });
  const context = await browser.newContext();
  const page = await context.newPage();

  // 拦截所有请求和响应
  page.on('request', req => {
    const url = req.url();
    const method = req.method();
    if (url.includes('woozooo.com') || url.includes('lanzou')) {
      console.log(`\n→ ${method} ${url}`);
    }
  });

  page.on('response', async res => {
    const url = res.url();
    const req = res.request();
    if (!url.includes('woozooo.com') && !url.includes('lanzou')) return;

    let body = '';
    try { body = await res.text(); } catch {}
    let postData = '';
    try { postData = req.postData() || ''; } catch {}

    logRequest(url, req.method(), postData, res, body);
  });

  // ── 1. 打开登录页 ──────────────────────────────────────────
  console.log('\n=== STEP 1: 打开登录页 ===');
  await page.goto('https://pc.woozooo.com/account.php?action=login', { waitUntil: 'networkidle' });
  await new Promise(r => setTimeout(r, 1500));

  // 截图看页面结构
  await page.screenshot({ path: 'login_page.png' });

  // 打印页面所有 input
  const inputs = await page.$$eval('input', els => els.map(e => ({
    name: e.name, type: e.type, id: e.id, placeholder: e.placeholder
  })));
  console.log('📋 页面 inputs:', JSON.stringify(inputs, null, 2));

  // ── 2. 填写表单并登录 ──────────────────────────────────────
  console.log('\n=== STEP 2: 填写登录表单 ===');

  // 找用户名输入框
  const uidInput = await page.$('input[name="uid"]') || await page.$('#username') || await page.$('input[type="text"]');
  const pwdInput = await page.$('input[name="pwd"]') || await page.$('#password') || await page.$('input[type="password"]');

  if (!uidInput || !pwdInput) {
    console.log('❌ 找不到登录输入框，打印页面HTML:');
    console.log((await page.content()).slice(0, 2000));
    await browser.close();
    return;
  }

  await uidInput.fill('');
  await uidInput.type(USERNAME, { delay: 80 });
  await pwdInput.fill('');
  await pwdInput.type(PASSWORD, { delay: 80 });

  await page.screenshot({ path: 'login_filled.png' });

  // 点击登录按钮
  const submitBtn = await page.$('input[type="submit"]') || await page.$('button[type="submit"]') || await page.$('.btn-login');
  if (submitBtn) {
    await submitBtn.click();
  } else {
    await page.keyboard.press('Enter');
  }

  await page.waitForNavigation({ waitUntil: 'networkidle', timeout: 15000 }).catch(() => {});
  await new Promise(r => setTimeout(r, 2000));
  await page.screenshot({ path: 'after_login.png' });
  console.log('📍 登录后 URL:', page.url());

  // ── 3. 获取 cookies ────────────────────────────────────────
  const cookies = await context.cookies();
  const cookieMap = Object.fromEntries(cookies.map(c => [c.name, c.value]));
  console.log('\n🍪 Cookies:', JSON.stringify(cookieMap, null, 2));

  const ylogin = cookieMap['ylogin'];
  if (!ylogin || ylogin === '0') {
    console.log('❌ 登录失败，ylogin=', ylogin);
    await page.screenshot({ path: 'login_failed.png' });
    await browser.close();
    return;
  }
  console.log('✅ 登录成功! ylogin=', ylogin);

  // ── 4. 访问 mydisk 抓文件夹列表请求 ───────────────────────
  console.log('\n=== STEP 3: 访问 mydisk 抓文件夹列表 ===');
  await page.goto('https://pc.woozooo.com/mydisk.php', { waitUntil: 'networkidle' });
  await new Promise(r => setTimeout(r, 3000));
  await page.screenshot({ path: 'mydisk.png' });

  // 打印页面里的 uid
  const pageHtml = await page.content();
  const uidMatch = pageHtml.match(/uid\s*[=:]\s*['"]*(\d+)/);
  console.log('📌 页面中的 uid:', uidMatch?.[1]);

  // ── 5. 输出汇总 ────────────────────────────────────────────
  console.log('\n\n========== 抓包汇总 ==========');
  console.log('Login captured:', !!captured.login);
  console.log('Folders captured:', captured.folders.length);
  console.log('Files captured:', captured.files.length);
  console.log('Cookies (for Python):', JSON.stringify(cookieMap));

  // 输出可直接用于 Python requests 的 cookie 字符串
  const cookieStr = cookies.map(c => `${c.name}=${c.value}`).join('; ');
  console.log('\n🐍 Python Cookie Header:');
  console.log(`Cookie: ${cookieStr}`);

  console.log('\n⏳ 保持浏览器 30 秒，可手动操作观察更多请求...');
  await new Promise(r => setTimeout(r, 30000));

  await browser.close();
}

main().catch(console.error);
