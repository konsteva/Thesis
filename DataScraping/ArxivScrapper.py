from bs4 import BeautifulSoup as bs
import requests
import csv

from tqdm import tqdm


class ArxivScrapper:
    def __init__(self):
        self.website_url = "https://arxiv.org/search/"
        self.search_details = "&searchtype=abstract&abstracts=show&order=announced_date_first&size=50"
        self.search_url = None
        self.keywords = None
        self.num_results = None
        self.soup = None
        self.data = None

    def _search(self, keywords):
        search_query = "?query="
        if not isinstance(keywords, list):
            search_query += f"%22{keywords}%22"
        else:
            for i, keyword in enumerate(keywords):
                # if more than one words in keyword
                if len(keyword.split(" ")) > 1:
                    keyword = "+".join(keyword.split(" "))
                search_query += f"%22{keyword}%22"
                if i != len(keywords) - 1:
                    search_query += "+%26%26+"

        self.search_url = self.website_url + search_query + self.search_details

        return self.search_url

    def _get_num_results(self):
        page_results = self.soup.find("main").find("h1", {"class": "title is-clearfix"}).text.split(" ")
        index = page_results.index('results')
        self.num_results = int(page_results[index - 1].replace(",", ""))

        return self.num_results

    def scrape(self, keywords):
        self.keywords = keywords
        search_url = self._search(keywords)

        response = requests.get(self.search_url)
        html = response.text
        self.soup = bs(html, 'html.parser')

        try:
            num_results = self._get_num_results()
        except ValueError:
            print(f"No results for keywords {keywords}")

            return

        page_counter = 0
        self.data = []
        for i in tqdm(range(0, num_results, 50)):
            url = search_url + "&start=" + str(i)
            response = requests.get(url)
            if response.status_code == 200:
                html = response.text
                soup = bs(html, 'html.parser')
            else:
                print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
                continue

            page_results = soup.find_all("li", {"class": "arxiv-result"})

            for result in page_results:
                submission_date_tag = result.find('p', class_='is-size-7')
                if submission_date_tag:
                    text = submission_date_tag.get_text(strip=True)

                    if "Submitted" in text:
                        submitted_index = text.index("Submitted")
                        date_text = text[submitted_index:].split(';')[0].replace('Submitted', '').strip()

                        self.data.append([date_text.split(", ")[-1]])

            page_counter += 1

    def save(self, filename):
        path = f"{filename}.csv"
        with open(path, 'w', encoding="utf-8", newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.data)
