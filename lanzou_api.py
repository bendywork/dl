"""
蓝奏云网盘 Python API
逆向自浏览器抓包，支持：登录、上传、文件夹列表、文件列表、直链、创建/删除文件夹、删除文件
"""

import re
import requests
from pathlib import Path

# 真实 UA（与抓包一致）
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36")

HEADERS = {
    "User-Agent": UA,
    "Referer":    "https://pc.woozooo.com/mydisk.php",
}

# 登录真实走 accounts.woozooo.com
ACCOUNTS_URL = "https://accounts.woozooo.com/accounts.php"
MYDISK_URL   = "https://pc.woozooo.com/mydisk.php"
DOUPLOAD_URL = "https://pc.woozooo.com/doupload.php"
FILEUP_URL   = "https://pc.woozooo.com/fileup.php"


class LanzouAPI:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._uid = None

    # ── 内部：构造带 uid 的 doupload URL ──────────────────────────────
    def _doupload_url(self):
        if self._uid:
            return f"{DOUPLOAD_URL}?uid={self._uid}"
        return DOUPLOAD_URL

    # ── 内部：从 vei 参数中获取（页面 JS 里有，直接省略也能跑大部分接口）─
    def _post(self, data: dict, timeout=15):
        return self.session.post(self._doupload_url(), data=data, timeout=timeout)

    # ------------------------------------------------------------------ #
    #  登录
    # ------------------------------------------------------------------ #
    def login(self, username: str, password: str) -> bool:
        """账号密码登录（走 accounts.woozooo.com）"""
        # 先访问登录页拿 PHPSESSID
        self.session.get(ACCOUNTS_URL, params={"action": "login", "ref": "pc.woozooo.com"}, timeout=15)

        data = {
            "action":   "login",
            "task":     "3",
            "ref":      "pc.woozooo.com",
            "username": username,
            "password": password,
        }
        resp = self.session.post(ACCOUNTS_URL, data=data, timeout=15)
        print(f"[*] 登录响应: {resp.text[:200]}")

        # 跟随跳转到 mydisk
        self.session.get(MYDISK_URL, timeout=15)
        self._extract_uid()

        ylogin = self.session.cookies.get("ylogin", "0")
        ok = ylogin not in ("0", "", None)
        print(f"[{'+'if ok else '-'}] 登录{'成功' if ok else '失败'}  ylogin={ylogin}  uid={self._uid}")
        return ok

    def login_with_cookie(self, cookies: dict) -> bool:
        """直接注入 cookie 跳过登录（推荐，用浏览器抓的 cookie）"""
        self.session.cookies.update(cookies)
        self.session.get(MYDISK_URL, timeout=15)
        self._extract_uid()
        ylogin = self.session.cookies.get("ylogin", "0")
        ok = ylogin not in ("0", "", None)
        print(f"[{'+'if ok else '-'}] Cookie 登录{'成功' if ok else '失败'}  ylogin={ylogin}  uid={self._uid}")
        return ok

    def _extract_uid(self):
        """从 mydisk 页面或 cookie 中提取数字 uid"""
        # 优先从 cookie ylogin 取（就是 uid）
        ylogin = self.session.cookies.get("ylogin", "")
        if ylogin and ylogin != "0":
            self._uid = ylogin
            return
        # fallback: 从页面 URL 参数 u= 提取
        resp = self.session.get(MYDISK_URL, timeout=15)
        m = re.search(r"[?&]u=(\d+)", resp.url)
        if m:
            self._uid = m.group(1)

    # ------------------------------------------------------------------ #
    #  文件夹操作
    # ------------------------------------------------------------------ #
    def get_folders(self, parent_id: int = -1) -> list:
        """获取文件夹列表，parent_id=-1 为根目录"""
        resp = self._post({"task": 47, "folder_id": parent_id})
        j = resp.json()
        print(f"[*] get_folders raw: {j}")
        folders = []
        for item in (j.get("text") or []):
            folders.append({
                "id":   item.get("fol_id") or item.get("id"),
                "name": item.get("name"),
            })
        return folders

    def create_folder(self, name: str, parent_id: int = -1, desc: str = "") -> int | None:
        """创建文件夹，返回新 id"""
        resp = self._post({
            "task":               2,
            "parent_id":          parent_id,
            "folder_name":        name,
            "folder_description": desc,
        })
        j = resp.json()
        if j.get("zt") == 1:
            fid = j.get("text")
            print(f"[+] 创建文件夹: {name}  id={fid}")
            return int(fid)
        print(f"[-] 创建文件夹失败: {j}")
        return None

    def delete_folder(self, folder_id: int) -> bool:
        """删除文件夹（进回收站）"""
        resp = self._post({"task": 3, "folder_id": folder_id})
        j = resp.json()
        ok = j.get("zt") == 1
        print(f"[{'+'if ok else '-'}] 删除文件夹 id={folder_id}: {j}")
        return ok

    # ------------------------------------------------------------------ #
    #  文件操作
    # ------------------------------------------------------------------ #
    def get_files(self, folder_id: int = -1) -> list:
        """获取文件列表（自动翻页）"""
        files, pg = [], 1
        while True:
            resp = self._post({"task": 5, "folder_id": folder_id, "pg": pg})
            j = resp.json()
            items = j.get("text") or []
            if not items:
                break
            for item in items:
                files.append({
                    "id":   item.get("id"),
                    "name": item.get("name_all") or item.get("name"),
                    "size": item.get("size"),
                    "time": item.get("time"),
                })
            # 翻页判断
            info = j.get("info")
            if isinstance(info, dict) and pg >= int(info.get("page", 1)):
                break
            elif not isinstance(info, dict):
                break
            pg += 1
        return files

    def delete_file(self, file_id: int) -> bool:
        """删除文件（进回收站）"""
        resp = self._post({"task": 6, "file_id": file_id})
        j = resp.json()
        ok = j.get("zt") == 1
        print(f"[{'+'if ok else '-'}] 删除文件 id={file_id}: {j}")
        return ok

    # ------------------------------------------------------------------ #
    #  上传
    # ------------------------------------------------------------------ #
    def upload_file(self, file_path: str, folder_id: int = -1) -> dict | None:
        """上传文件，返回 {"id", "name", "share_url"} 或 None"""
        path = Path(file_path)
        if not path.exists():
            print(f"[-] 文件不存在: {file_path}")
            return None

        print(f"[*] 上传: {path.name} → folder_id={folder_id}")
        with open(path, "rb") as f:
            files = {"upload_file": (path.name, f, "application/octet-stream")}
            data  = {
                "task":           "1",
                "vie":            "2",
                "ve":             "2",
                "id":             "WU_FILE_0",
                "name":           path.name,
                "folder_id_bb_n": str(folder_id),
            }
            # 上传需要正确的 Referer
            headers = {**HEADERS, "Referer": "https://pc.woozooo.com/mydisk.php"}
            resp = self.session.post(FILEUP_URL, data=data, files=files,
                                     headers=headers, timeout=120)

        print(f"[*] 上传响应: {resp.text[:300]}")
        try:
            j = resp.json()
        except Exception:
            print("[-] 响应不是 JSON")
            return None

        if j.get("zt") == 1:
            info = j.get("text", {})
            result = {
                "id":        info.get("id") or info.get("file_id"),
                "name":      path.name,
                "share_url": info.get("f_id") or "",
            }
            print(f"[+] 上传成功: {result}")
            return result
        print(f"[-] 上传失败: {j}")
        return None

    # ------------------------------------------------------------------ #
    #  获取直链
    # ------------------------------------------------------------------ #
    def get_direct_link(self, file_id: int, pwd: str = "") -> str | None:
        """通过 file_id 获取下载直链"""
        resp = self._post({"task": 22, "file_id": file_id})
        j = resp.json()
        print(f"[*] task=22 raw: {j}")
        if j.get("zt") != 1:
            print(f"[-] 获取分享链接失败: {j}")
            return None
        share_url = j["text"]
        return self._resolve_share_url(share_url, pwd)

    def _resolve_share_url(self, share_url: str, pwd: str = "") -> str | None:
        """分享链接 → 真实直链"""
        resp = self.session.get(share_url, timeout=15)
        html = resp.text

        sign = ""
        for pattern in [
            r"var\s+signs?\s*=\s*['\"]([^'\"]+)['\"]",
            r"'sign'\s*:\s*'([^']+)'",
            r'"sign"\s*:\s*"([^"]+)"',
        ]:
            m = re.search(pattern, html)
            if m:
                sign = m.group(1)
                break

        if not sign:
            print(f"[-] 无法从分享页提取 sign，URL={share_url}")
            return None

        ajax_m = re.search(r"(https?://[^/\"']+)/ajaxm\.php", html)
        ajax_base = ajax_m.group(1) if ajax_m else "https://www.lanzouo.com"

        post_data = {"action": "downprocess", "sign": sign, "p": pwd}
        if not pwd:
            post_data["ves"] = "1"

        resp2 = self.session.post(
            f"{ajax_base}/ajaxm.php", data=post_data,
            headers={**HEADERS, "Referer": share_url}, timeout=15
        )
        j2 = resp2.json()
        if j2.get("zt") != 1:
            print(f"[-] ajaxm 解析失败: {j2}")
            return None

        download_url = f"{j2['dom']}/file/{j2['url']}"
        resp3 = self.session.get(download_url, allow_redirects=True, timeout=15)
        final = resp3.url
        print(f"[+] 直链: {final}")
        return final


# ------------------------------------------------------------------ #
#  主程序
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    api = LanzouAPI()

    # ── 方式1：用浏览器抓包得到的 cookie 直接登录（最稳定）──────────
    COOKIES = {
        "PHPSESSID":    "02im10qebtlfsqvbksjua7h904hj561b",
        "phpdisk_info": "VmcFNVEyV25QawFhCmdQAwBkDAdcNFA%2BV2YCZgU2VGEEOVdiVjBRag48BF0OZQFjUzVXNABrXWgBOglpDmwLbFY2BTRRM1djUGQBYQoyUDoAYww5XDVQZlcxAjQFZVRuBGRXZVY8UT4ObwRmDl0BalM6V2cAbV0%2FATUJYA4%2FCz1WZQU3",
        "uag":          "148d229c2349f686b51aedf0ab0cd4ef",
        "ylogin":       "5215907",
    }
    ok = api.login_with_cookie(COOKIES)

    # ── 方式2：账号密码登录（备用）────────────────────────────────────
    # ok = api.login("19245797470", "99love")

    if not ok:
        print("登录失败，退出")
        exit(1)

    # ── 文件夹列表 ──────────────────────────────────────────────────
    print("\n=== 文件夹列表 ===")
    folders = api.get_folders(-1)
    for f in folders:
        print(f"  [{f['id']}] {f['name']}")

    # ── 文件列表 ────────────────────────────────────────────────────
    print("\n=== 文件列表 ===")
    files = api.get_files(-1)
    for f in files:
        print(f"  [{f['id']}] {f['name']}  {f['size']}  {f['time']}")

    # ── 上传测试文件 ────────────────────────────────────────────────
    print("\n=== 上传测试 ===")
    result = api.upload_file(
        "/Volumes/Data/libs/npms/global/lib/node_modules/cloakbrowser/README.md",
        folder_id=-1
    )

    # ── 如果上传成功，获取直链 ─────────────────────────────────────
    if result and result.get("id"):
        print("\n=== 获取直链 ===")
        api.get_direct_link(result["id"])

    # ── 创建文件夹示例 ─────────────────────────────────────────────
    # fid = api.create_folder("测试文件夹")

    # ── 删除文件示例 ───────────────────────────────────────────────
    # api.delete_file(result["id"])

    # ── 删除文件夹示例 ─────────────────────────────────────────────
    # api.delete_folder(fid)
