import json
from playwright.async_api import async_playwright
import logging
import time
import pandas as pd
from tqdm.asyncio import tqdm
import asyncio
import httpx
from typing import Optional, Tuple, Any
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)


class AsyncWbParser:
    URL = "https://www.wildberries.ru"
    SEARCH_URL = "https://www.wildberries.ru/__internal/u-search/exactmatch/ru/common/v18/search"

    DEFAULT_PARAMS = {
        'ab_testing': 'false',
        'appType': '1',
        'curr': 'rub',
        'dest': '-1257786',
        'hide_dtype': '9',
        'hide_vflags': '4294967296',
        'lang': 'ru',
        'page': '1',
        'query': 'dsaila pro',
        'resultset': 'catalog',
        'sort': 'popular',
        'spp': '30',
        'suppressSpellcheck': 'false',
    }

    def __init__(self):
        self.client: Optional[httpx.AsyncClient] = None
        self.cookies = {}
        self.user_agent = None

    @classmethod
    async def create(cls) -> "AsyncWbParser":
        self = cls()
        self.cookies, self.user_agent = await self._get_token_static()
        limits = httpx.Limits(max_keepalive_connections=16, max_connections=30)
        self.client = httpx.AsyncClient(http2=True, limits=limits, headers={
                                        "User-Agent": self.user_agent}, cookies=self.cookies, timeout=10.0)
        return self

    async def close(self):
        if self.client:
            await self.client.aclose()

    async def search_raw_data(self, query: str, page: int, retry: int = 3) -> Optional[dict]:
        params = self.DEFAULT_PARAMS.copy()
        params['query'] = query
        params['page'] = str(page)
        for attempt in range(1, retry + 1):
            try:
                resp = await self.client.get(self.SEARCH_URL, params=params)
            except (httpx.RequestError, httpx.TimeoutException) as e:
                logger.warning(
                    "search_raw_data request error: %s (attempt %d)", e, attempt)
                await asyncio.sleep(0.5 * attempt)
                continue

            if resp.status_code == 498:
                logger.warning(
                    "token expired, refreshing (attempt %d)", attempt)
                await self._refresh_token()
                continue
            if resp.status_code != 200:
                logger.error("search_raw_data HTTP %d", resp.status_code)
                return None
            try:
                return resp.json()
            except Exception as e:
                logger.error("JSON parse error: %s", e)
                return None
        return None

    async def search_all_data(self, query: str) -> list[dict]:
        page = 1
        all_products: list[dict] = []
        empty_pages_in_a_row = 0
        max_empty_pages = 20
        with tqdm(desc=f"Сбор товаров по '{query}'", unit="page") as pbar:
            while True:
                raw_data = await self.search_raw_data(query, page)
                products = raw_data.get("products", []) if raw_data else []

                if len(products) == 0:
                    empty_pages_in_a_row += 1
                else:
                    empty_pages_in_a_row = 0

                if empty_pages_in_a_row >= max_empty_pages:
                    break

                all_products.extend(products)
                page += 1
                pbar.update(1)
                await asyncio.sleep(0.2)
        return all_products

    def extract_products(self, raw_data: list[dict]) -> list[dict]:
        extracted_data = []
        for prod in raw_data:
            sku = prod.get("id")
            
            sizes = prod.get("sizes", [])
            first_available_size = next((s for s in sizes if s.get("wh", 0) > 0), None)
            price_raw = None
            if first_available_size:
                price_raw = first_available_size.get('price', {}).get('product') \
                            or first_available_size.get('price', {}).get('basic')
            all_sizes = ", ".join([str(s.get("name"))
                                  for s in sizes if s.get("name")])
            item = {
                "link": f"https://www.wildberries.ru/catalog/{sku}/detail.aspx",
                "sku": sku,
                "name": prod.get('name'),
                "price": price_raw / 100,
                "sizes": all_sizes,
                "rating": prod.get('reviewRating'),
                "feedbacks": prod.get('feedbacks'),
                "seller_name": prod.get('supplier'),
                "seller_link": f"https://www.wildberries.ru/seller/{prod.get('supplierId')}",
                "total_stocks": prod.get('totalQuantity'),
                "pics_count": prod.get('pics'),

                "description": "",
                "characteristics": {},
                "country": ""
            }
            extracted_data.append(item)
        return extracted_data

    async def get_detail_product(self, sku: int) -> Tuple[Optional[dict], str]:
        vol = sku // 100000
        part = sku // 1000
        base_basket = self._get_basket_id(sku)
        offsets = list(dict.fromkeys([i for j in range(15) for i in (j, -j)]))

        for offset in offsets:
            basket_num = base_basket + offset
            if basket_num < 1:
                continue
            basket_str = f"{basket_num:02d}"
            url = f"https://basket-{basket_str}.wbcontent.net/vol{vol}/part{part}/{sku}/info/ru/card.json"

            res_json, found_basket = await self._fetch_basket(url, basket_str)
            if res_json:
                return res_json, found_basket

        return None, ""

    def generate_image_urls(self, sku: int, pics_count: int, basket_str: str) -> str:
        if not pics_count and basket_str is None:
            return ""

        vol = sku // 100000
        part = sku // 1000
        base_url = f"https://basket-{basket_str}.wbbasket.ru/vol{vol}/part{part}/{sku}/images/big"
        image_links = [
            f"{base_url}/{i}.webp" for i in range(1, pics_count + 1)]
        return ", ".join(image_links)

    def extract_detail_info(self, detail_json: dict) -> dict:
        if not detail_json:
            return {}

        description = detail_json.get("description", "")
        country = "Не указано"
        structured_data = []
        for group in detail_json.get("grouped_options", []):
            group_name = group.get("group_name")
            options_list = []
            for opt in group.get("options", []):
                name = opt.get("name")
                value = opt.get("value")
                if name == 'Страна производства':
                    country = value
                options_list.append(f"{name}: {value}")

            group_string = f"[{group_name}]\n" + "\n".join(options_list)
            structured_data.append(group_string)

        characteristics = "\n\n".join(structured_data)

        return {
            "description": description,
            "characteristics": characteristics,
            "country": country
        }

    async def _fetch_basket(self, url: str, basket_str: str) -> Tuple[Optional[dict], Optional[str]]:
        try:
            response = await self.client.get(url, timeout=1)
            if response.status_code == 200:
                return response.json(), basket_str
        except Exception:
            pass
        return None, None

    def _get_basket_id(self, sku: int) -> str:
        vol = sku // 100000
        basket_ranges = [
            (143, 1), (287, 2), (431, 3), (719, 4), (1007, 5),
            (1061, 6), (1115, 7), (1169, 8), (1313, 9), (1601, 10),
            (1655, 11), (1919, 12), (2045, 13), (2189, 14), (2405, 15),
            (2621, 16), (2837, 17), (3053, 18), (3269, 19), (3485, 20),
            (3701, 21), (3917, 22), (4133, 23), (4349, 24), (4565, 25),
            (4781, 26), (5099, 27), (5414, 28), (5729, 29), (6044, 30),
            (6359, 31), (6674, 32), (6989, 33), (7304, 34), (7619, 35)
        ]
        for limit, basket in basket_ranges:
            if vol <= limit:
                return basket
        return 36

    @staticmethod
    async def _get_token_static() -> Tuple[dict, str]:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                channel="chrome",
                headless=False,
                args=['--disable-blink-features=AutomationControlled']
            )
            context = await browser.new_context()
            page = await context.new_page()
            await page.goto(AsyncWbParser.URL, wait_until="networkidle")

            await page.wait_for_event("framenavigated")

            cookies_list = await context.cookies()
            cookies = {c['name']: c['value'] for c in cookies_list}

            user_agent = await page.evaluate("() => navigator.userAgent")

            await browser.close()
            return cookies, user_agent

    async def _refresh_token(self):
        self.cookies, self.user_agent = await self._get_token_static()
        if self.client:
            self.client.headers.update({"User-Agent": self.user_agent})
            self.client.cookies = httpx.Cookies(self.cookies)


async def main():
    parser = await AsyncWbParser.create()
    try:
        raw_products = await parser.search_all_data(
            "пальто из натуральной шерсти"
        )
        logger.info(f"Найдено товаров: {len(raw_products)}")
    except Exception as e:
        logger.error(f"Ошибка при поиске товаров: {e}")
        await parser.close()
        return
    try:
        products = parser.extract_products(raw_products)
        semaphore = asyncio.Semaphore(20)

        async def worker(p: dict):
            async with semaphore:
                await asyncio.sleep(0.1)
                detail_product, basket_str = await parser.get_detail_product(p['sku'])
                if detail_product:
                    p.update(parser.extract_detail_info(detail_product))
                    p["image_urls"] = parser.generate_image_urls(
                        p["sku"], p["pics_count"], basket_str)
        start = time.perf_counter()

        await tqdm.gather(*(worker(p) for p in products), desc="Обработка товаров", total=len(products))
        end = time.perf_counter()
        logger.info(f"Общее время выполнения: {end - start:.2f} секунд")
        logger.info(f'Обработано товаров: {len(products)}')
    finally:
        await parser.close()
    column_mapping = {
        "link": "Ссылка на товар",
        "sku": "Артикул",
        "name": "Название",
        "price": "Цена",
        "description": "Описание",
        "image_urls": "Ссылки на изображения через запятую",
        "characteristics": "Все характеристики с сохранением их структуры",
        "seller_name": "Название селлера",
        "seller_link": "Ссылка на селлера",
        "sizes": "Размеры товара через запятую",
        "total_stocks": "Остатки по товару (число)",
        "rating": "Рейтинг",
        "feedbacks": "Количество отзывов",
        "country": "Страна производства"
    }
    df = pd.DataFrame(products).rename(columns=column_mapping)
    col_to_export = list(column_mapping.values())
    df[col_to_export].to_excel("wb_detailed_data.xlsx", index=False)
    filtered_df = df[
        (df["Рейтинг"] >= 4.5) &
        (df["Цена"] <= 10000) &
        (df["Страна производства"].str.contains("Россия", case=False, na=False))
    ]

    filtered_df[col_to_export].to_excel("filtered_catalog.xlsx", index=False)


if __name__ == "__main__":
    asyncio.run(main())
