from wbparse_async import WBProduct, WBSize, WBPrice

def test_wb_product_price_calc():
    # Проверяем, что цена берется из первого доступного размера с остатком
    product = WBProduct(
        id=123,
        sizes=[
            WBSize(name="S", wh=0, price=WBPrice(product=10000)),
            WBSize(name="M", wh=10, price=WBPrice(product=500000)) 
        ]
    )
    assert product.get_price() == 5000.0
    