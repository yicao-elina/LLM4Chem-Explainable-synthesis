import asyncio
import json
import re
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

async def scrape_material():
    """
    Scrapes the material synthesis data from 2dmat.chemdx.org.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        try:
            # Navigate to the website
            await page.goto("https://2dmat.chemdx.org/data")

            # Wait for the table to load
            await page.wait_for_selector("table")

            # Get the page content
            content = await page.content()

            # Parse the content with BeautifulSoup
            soup = BeautifulSoup(content, "html.parser")

            # Find the table
            table = soup.find("table")

            # Get the table headers
            headers = [header.text for header in table.find_all("th")]

            # Get the table rows
            rows = table.find("tbody").find_all("tr")

            # Extract the data
            data = []
            for row in rows:
                cols = row.find_all("td")
                
                # Clean up the ID field
                id_text = cols[0].text.strip()
                id_match = re.search(r"^[^\s]+", id_text)
                if id_match:
                    id_text = id_match.group(0)

                # Get the rest of the columns
                other_cols = [ele.text.strip() for ele in cols[1:]]
                
                # Create the data dictionary
                row_data = {headers[0]: id_text}
                for i, header in enumerate(headers[1:]):
                    row_data[header] = other_cols[i]
                
                data.append(row_data)

            return data

        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        finally:
            await browser.close()

if __name__ == "__main__":
    scraped_data = asyncio.run(scrape_material())

    if scraped_data:
        # Save the data to a JSON file
        with open("materials.json", "w") as f:
            json.dump(scraped_data, f, indent=2)
        print("Data scraped and saved to materials.json")