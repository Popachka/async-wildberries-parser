import json
import pytest
from pydantic import ValidationError
from wbparse_async import WBProduct, AsyncWbParser


@pytest.mark.asyncio
async def test_current_wb_api_structure():
    parser = AsyncWbParser()

    products = await parser.search_products_catalog("пальто", page=1)
        
    assert len(products) > 0, "API вернуло 0 товаров. Возможно, изменились параметры запроса или URL"
    
    assert isinstance(products[0], WBProduct)