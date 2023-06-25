import json
import logging
import os.path
from time import sleep
from urllib.parse import urljoin
from slugify import slugify

import requests
from datasets import load_dataset
from lxml.html.soupparser import fromstring


class PapersWithCodeDatasetScraper:
    def parse_dataset_page(self, dataset_page_url):
        dataset_page = requests.get(dataset_page_url)
        tree = fromstring(dataset_page.text)
        dataset_loaders = tree.xpath("//a[@class='code-table-link']/@href")
        for ds_loader in dataset_loaders:
            if 'huggingface' in ds_loader:
                huggingface_dataset_name = ds_loader.replace("https://huggingface.co/datasets/", '')
                try:
                    dataset = load_dataset(huggingface_dataset_name, streaming=True)
                    if next(iter(dataset)):
                        return (ds_loader, huggingface_dataset_name, tree)
                    break
                except Exception as e:
                    logging.error(e)
                    break

    def scrape(self, url, max_pages=1):
        if not os.path.isdir('papers_with_code_data/images'):
            os.makedirs('papers_with_code_data/images')
        page = 3
        base_url = 'https://paperswithcode.com'
        page_link = url + f'&page={page}'
        for i in range(page, max_pages + 1):
            page_content = requests.get(page_link)
            tree = fromstring(page_content.text)
            hrefs = tree.xpath("//div[@class='dataset-wide-box']//a/@href")
            for k, href in enumerate(hrefs):
                if os.path.isfile(f'papers_with_code_data/images/{i}-{k}-{slugify(href)}.json'):
                    continue
                sleep(3)
                print(f'Scraping {href}')
                dataset_page_url = urljoin(base_url, href)
                result = self.parse_dataset_page(dataset_page_url)
                if result:
                    (parsed_hugging_face_url, huggingface_dataset_name, dataset_page_tree) = result
                    benchmarks = dataset_page_tree.xpath("//table[@id='benchmarks-table']//tr//@onclick")
                    j = []
                    for benchmark in benchmarks:
                        benchmark_url = urljoin(base_url, benchmark.replace("window.location='", '').replace("';", ''))
                        benchmark_page_text = requests.get(benchmark_url).text
                        benchmark_page = fromstring(benchmark_page_text)
                        table_data = benchmark_page.xpath("//script[@id='evaluation-table-data']//text()")
                        sota_page_details = benchmark_page.xpath("//script[@id='sota-page-details']//text()")
                        table_data = json.loads(table_data[0])
                        sota_page_details = json.loads(sota_page_details[0])
                        j.append({
                            'hugging_face_dataset_name': huggingface_dataset_name,
                            'hugging_face': parsed_hugging_face_url,
                            'dataset_page_url': dataset_page_url,
                            'benchmark_url': benchmark_url,
                            'table': table_data,
                            'sota': sota_page_details
                        })
                    with open(f'papers_with_code_data/images/{i}-{k}-{slugify(href)}.json', 'w') as f:
                        json.dump(j, f)


if __name__ == '__main__':
    PapersWithCodeDatasetScraper().scrape('https://paperswithcode.com/datasets?mod=images', 5)
