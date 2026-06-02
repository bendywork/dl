/**
 * 用 Node.js 原生 https + FormData 模拟蓝奏云上传
 * cookie 来自 capture_upload.mjs 保存的 cookies.json
 */
import { readFileSync } from 'fs';
import { request } from 'https';
import { basename } from 'path';

const FILE_PATH = '/Volumes/Data/download/bendy-jizhang-v0.6.1-arm64-v8a.apk';
const cookies = JSON.parse(readFileSync('./cookies.json', 'utf8'));

const COOKIE_STR = Object.entries(cookies).map(([k,v]) => `${k}=${v}`).join('; ');
const UID = cookies.ylogin;

function uploadFile(filePath, folderId = -1) {
  return new Promise((resolve, reject) => {
    const fileName = basename(filePath);
    const fileData = readFileSync(filePath);

    const boundary = '----WebKitFormBoundary' + Math.random().toString(36).slice(2);

    const fields = {
      task: '1',
      vie: '2',
      ve: '2',
      id: 'WU_FILE_0',
      name: fileName,
      folder_id_bb_n: String(folderId),
    };

    // 构造 multipart body
    let bodyParts = [];
    for (const [key, val] of Object.entries(fields)) {
      bodyParts.push(
        `--${boundary}\r\n` +
        `Content-Disposition: form-data; name="${key}"\r\n\r\n` +
        `${val}\r\n`
      );
    }
    // 文件字段
    const fileHeader =
      `--${boundary}\r\n` +
      `Content-Disposition: form-data; name="upload_file"; filename="${fileName}"\r\n` +
      `Content-Type: application/octet-stream\r\n\r\n`;
    const fileFooter = `\r\n--${boundary}--\r\n`;

    const headerBuf  = Buffer.from(bodyParts.join('') + fileHeader, 'utf8');
    const footerBuf  = Buffer.from(fileFooter, 'utf8');
    const totalLen   = headerBuf.length + fileData.length + footerBuf.length;
    const body       = Buffer.concat([headerBuf, fileData, footerBuf]);

    const options = {
      hostname: 'pc.woozooo.com',
      path: '/fileup.php',
      method: 'POST',
      headers: {
        'Content-Type': `multipart/form-data; boundary=${boundary}`,
        'Content-Length': totalLen,
        'Cookie': COOKIE_STR,
        'Referer': 'https://pc.woozooo.com/mydisk.php',
        'Origin': 'https://pc.woozooo.com',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36',
        'Accept': '*/*',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'X-Requested-With': 'XMLHttpRequest',
      },
    };

    console.log(`上传: ${fileName} (${(fileData.length/1024/1024).toFixed(2)} MB) → folder_id=${folderId}`);
    console.log(`Cookie: ylogin=${cookies.ylogin}`);

    const req = request(options, res => {
      let data = '';
      res.on('data', chunk => { data += chunk; process.stdout.write('.'); });
      res.on('end', () => {
        console.log('\nStatus:', res.statusCode);
        console.log('Response:', data);
        resolve({ status: res.statusCode, body: data });
      });
    });

    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

// 也测一下 doupload task=47 文件夹列表
function getFolders() {
  return new Promise((resolve, reject) => {
    const body = `task=47&folder_id=-1`;
    const options = {
      hostname: 'pc.woozooo.com',
      path: `/doupload.php?uid=${UID}`,
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Content-Length': Buffer.byteLength(body),
        'Cookie': COOKIE_STR,
        'Referer': 'https://pc.woozooo.com/mydisk.php',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36',
        'X-Requested-With': 'XMLHttpRequest',
      },
    };
    const req = request(options, res => {
      let data = '';
      res.on('data', c => data += c);
      res.on('end', () => resolve(JSON.parse(data)));
    });
    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

async function main() {
  console.log('=== 测试 task=47 文件夹列表 ===');
  const folders = await getFolders();
  console.log('folders:', JSON.stringify(folders, null, 2));

  console.log('\n=== 上传文件 ===');
  await uploadFile(FILE_PATH, -1);
}

main().catch(console.error);
