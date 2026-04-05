
import requests
import pandas as pd
import json
from time import sleep
from tqdm import tqdm

class MC2DDataScraper:
    def __init__(self):
        self.base_url = "https://mcxd-api.materialscloud.org/mc2d/pbe-v1"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MC2D-Research-Scraper/1.0'
        })
    
    def get_material_ids(self):
        """Get list of all material IDs from overview"""
        try:
            response = self.session.get(f"{self.base_url}/overview")
            response.raise_for_status()
            data = response.json()
            
            # Extract material IDs from the list of dictionaries
            material_ids = [item.get('id') for item in data if item.get('id')]
            print(f"Found {len(material_ids)} materials")
            return material_ids
            print(f"Found {len(material_ids)} materials")
            return material_ids
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching overview: {e}")
            return []
    
    def get_material_data(self, material_id):
        """Get detailed data for a specific material"""
        try:
            url = f"{self.base_url}/core_base/{material_id}"
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {material_id}: {e}")
            return None
    
    def scrape_all_materials(self, delay=0.5):
        """Scrape all materials with progress bar"""
        material_ids = self.get_material_ids()
        all_data = []
        
        for mat_id in tqdm(material_ids, desc="Scraping materials"):
            data = self.get_material_data(mat_id)
            if data:
                all_data.append(data)
            sleep(delay)  # Be polite to the server
        
        return all_data
    
    def save_to_csv(self, data, filename='mc2d_materials.csv'):
        """Convert to DataFrame and save as CSV"""
        df = pd.json_normalize(data)
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
        return df
    
    def save_to_json(self, data, filename='mc2d_materials.json'):
        """Save raw JSON data"""
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Data saved to {filename}")


# Usage
if __name__ == "__main__":
    scraper = MC2DDataScraper()
    
    # Scrape all materials
    materials_data = scraper.scrape_all_materials(delay=0.5)
    
    # Save in both formats
    df = scraper.save_to_csv(materials_data)
    scraper.save_to_json(materials_data)
    
    # Display basic info
    print(f"\nTotal materials scraped: {len(materials_data)}")
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst few rows:\n{df.head()}")
