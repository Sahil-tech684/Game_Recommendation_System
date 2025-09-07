import scrapy
import json
from scrapy.crawler import CrawlerProcess

class GamesSpider(scrapy.Spider):
    name = 'games'
    start_urls = ['https://rawg.io/api/games?page=1&page_size=40&filter=true&comments=true&key=c542e67aec3a4340908f9de9e86038af']
    custom_settings = {
        'FEED_FORMAT': 'csv',
        'FEED_URI': r'Data/GamesData.csv',
        'concurrent_requests': 30
    }

    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36'
    }

    def parse(self, response):
        data = json.loads(response.text)
        results = data.get('results', [])
        for result in results:
            genre_names = [genre['name'] for genre in result.get('genres', [])]
            poster = result.get('background_image')
            id = result.get('id')
            url = f'https://rawg.io/games/{id}'
            yield scrapy.Request(url, callback=self.parse_page, headers=self.HEADERS, meta={'id': id, 'genre': genre_names, 'poster' : poster})
        next_page = data.get('next')
        if next_page:
            yield scrapy.Request(url=next_page, callback=self.parse)

    def parse_page(self, response):
        yield {
            'id': response.meta['id'],
            'name': response.css('div.game__head h1.heading::text').get(),
            'about': response.css('div.game__about > div.game__about-text ::text').getall(),
            'genre': response.meta['genre'],
            'publishers' : response.css('div.game__meta-block:contains("Publisher") ::text').getall()[1:],
            'developers' : response.css('div.game__meta-block:contains("Developer") ::text').getall()[1:],
            'tags' : response.css('div.game__meta-block:contains("Tags") ::text').getall()[1:],
            'other_games_series' : response.css('div.game__meta-block:contains("Other games in the series") ::text').getall()[1:],
            'website' : response.css('div.game__meta-block:contains("Website") ::text').getall()[1:],
            'poster' : response.meta['poster']
        }

if __name__ == '__main__':
    process = CrawlerProcess()
    process.crawl(GamesSpider)
    process.start()
