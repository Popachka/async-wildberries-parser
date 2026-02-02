import json
from playwright.async_api import async_playwright
import logging
import time
import pandas as pd
from tqdm.asyncio import tqdm
import asyncio
import httpx
from typing import Optional, Tuple, Any
from pydantic import BaseModel, Field, field_validator
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)

# Data models


class WBPrice(BaseModel):
    product: int = 0
    basic: int = 0


class WBSize(BaseModel):
    name: str = ""
    wh: int = 0
    price: Optional[WBPrice] = None


class WBProduct(BaseModel):
    """Model product from raw search"""
    id: int
    name: str = "Без названия"
    supplier: str = "Неизвестный продавец"
    supplierId: int = 0
    reviewRating: float = 0.0
    feedbacks: int = 0
    totalQuantity: int = 0
    pics: int = 0
    sizes: list[WBSize] = Field(default_factory=list)

    description: str = ""
    characteristics: str = ""
    country: str = "Не указано"
    image_urls: str = ""
    
    def generate_images(self, basket_str: str):
        """Генерирует строку со ссылками на фото"""
        if not self.pics or not basket_str:
            return ""
        vol = self.id // 100000
        part = self.id // 1000
        base_url = f"https://basket-{basket_str}.wbbasket.ru/vol{vol}/part{part}/{self.id}/images/big"
        links = [f"{base_url}/{i}.webp" for i in range(1, self.pics + 1)]
        self.image_urls = ", ".join(links)
    
    @property
    def sku(self) -> int:
        return self.id

    def get_price(self) -> float:
        for size in self.sizes:
            if size.wh > 0 and size.price:
                return (size.price.product or size.price.basic) / 100
        return 0.0

    def get_sizes_str(self) -> str:
        return ", ".join([s.name for s in self.sizes if s.name])
    def to_excel_dict(self):
        return {
            "link": f"https://www.wildberries.ru/catalog/{self.id}/detail.aspx",
            "sku": self.id,
            "name": self.name,
            "price": self.get_price(),
            "description": self.description,
            "image_urls": self.image_urls,
            "characteristics": self.characteristics,
            "seller_name": self.supplier,
            "seller_link": f"https://www.wildberries.ru/seller/{self.supplierId}",
            "sizes": self.get_sizes_str(),
            "total_stocks": self.totalQuantity,
            "rating": self.reviewRating,
            "feedbacks": self.feedbacks,
            "country": self.country
        }

class WBOption(BaseModel):
    name: str = ""
    value: str = ""


class WBGroupedOption(BaseModel):
    group_name: str = ""
    options: list[WBOption] = Field(default_factory=list)


class WBDetailProduct(BaseModel):
    description: str = ""
    grouped_options: list[WBGroupedOption] = Field(default_factory=list)


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
        limits = httpx.Limits(
            max_keepalive_connections=30, max_connections=1000)
        self.client = httpx.AsyncClient(http2=True, limits=limits, headers={
                                        "User-Agent": self.user_agent}, cookies=self.cookies, timeout=10.0)
        return self

    async def close(self):
        if self.client:
            await self.client.aclose()

    async def search_products_catalog(self, query: str, page: int, retry: int = 3) -> list[WBProduct]:
        params = self.DEFAULT_PARAMS.copy()
        params.update({"query": query, "page": str(page)})

        for attempt in range(1, retry + 1):
            try:
                resp = await self.client.get(self.SEARCH_URL, params=params)
                if resp.status_code == 498:
                    logger.warning(
                        "token expired, refreshing (attempt %d)", attempt)
                    await self._refresh_token()
                    continue
                if resp.status_code != 200:
                    return []
                data = resp.json()
                products = data.get("products", [])
                return [WBProduct.model_validate(p) for p in products]
            except Exception as e:
                logger.warning(f"Attempt {attempt} failed: {e}")
                await asyncio.sleep(0.5)
        return []

    async def search_all_products(self, query: str, max_pages: Optional[int] = None) -> list[WBProduct]:
        page, all_products,  = 1, []
        empty_pages_in_a_row, max_empty = 0, 5
        with tqdm(desc=f"Сбор товаров по '{query}'", unit="page") as pbar:
            while True:
                products = await self.search_products_catalog(query, page)
                
                if len(products) == 0:
                    empty_pages_in_a_row += 1
                    if empty_pages_in_a_row >= max_empty:
                        logger.info(f"Поиск завершен: {max_empty} пустых страниц подряд.")
                        break
                else:
                    empty_pages_in_a_row = 0
                    all_products.extend(products)
                page += 1
                pbar.update(1)
                if max_pages and max_pages == page:
                    break
                await asyncio.sleep(0.2)
        return all_products

    async def get_detail_product(self, sku: int) -> tuple[Optional[WBDetailProduct], str]:
        vol, part = sku // 100000, sku // 1000
        base_basket = self._get_basket_id(sku)
        offsets = list(dict.fromkeys([i for j in range(15) for i in (j, -j)]))

        for offset in offsets:
            basket_str = f"{base_basket + offset:02d}"
            url = f"https://basket-{basket_str}.wbcontent.net/vol{vol}/part{part}/{sku}/info/ru/card.json"
            try:
                res = await self.client.get(url, timeout=1)
                if res.status_code == 200:
                    return WBDetailProduct.model_validate(res.json()), basket_str
            except:
                continue
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
        products = await parser.search_all_products(
            "пальто из натуральной шерсти",
            5
        )
        logger.info(f"Найдено товаров: {len(products)}")
    except Exception as e:
        logger.error(f"Ошибка при поиске товаров: {e}")
        await parser.close()
        return
    try:
        semaphore = asyncio.Semaphore(20)

        async def worker(product: WBProduct):
            async with semaphore:
                await asyncio.sleep(0.1)
                detail, b_str = await parser.get_detail_product(product.sku)
                if detail:
                    product.generate_images(b_str)
                    product.description = detail.description
                    chars = []
                    for group in detail.grouped_options:
                        opts = [f"{o.name}: {o.value}" for o in group.options]
                        if opts:
                            chars.append(f"[{group.group_name}]\n" + "\n".join(opts))
                            for o in group.options:
                                if "Страна" in o.name:
                                    product.country = o.value
                    
                    product.characteristics = "\n\n".join(chars)
        start = time.perf_counter()
        await tqdm.gather(*(worker(p) for p in products), desc="Обработка товаров", total=len(products))
        end = time.perf_counter()
        logger.info(f"Общее время выполнения: {end - start:.2f} секунд")
        logger.info(f'Обработано товаров: {len(products)}')
        final_data = [p.to_excel_dict() for p in products]

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
    df = pd.DataFrame(final_data).rename(columns=column_mapping)
    print(df)
    col_to_export = list(column_mapping.values())
    df[col_to_export].to_excel("wb_detailed_data.xlsx", index=False)
    filtered_df = df[
        (df["Рейтинг"] >= 4.5) &
        (df["Цена"] <= 10000) &
        (df["Страна производства"].str.contains("Россия", case=False, na=False))
    ]
    if not filtered_df.empty:
        filtered_df[col_to_export].to_excel("filtered_catalog.xlsx", index=False)


if __name__ == "__main__":
    asyncio.run(main())
