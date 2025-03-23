import os
import json
import logging
import trafilatura
import random
import time
import requests
from bs4 import BeautifulSoup

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_website_text_content(url):
    """
    Extract clean text content from a website
    
    Args:
        url: URL of the website to scrape
        
    Returns:
        Extracted text content
    """
    try:
        logger.info(f"Scraping content from: {url}")
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded)
        return text
    except Exception as e:
        logger.error(f"Error extracting content from {url}: {str(e)}")
        return None

def scrape_plant_disease_data():
    """
    Scrape plant disease data from various reliable sources and save to JSON
    """
    disease_data = []
    
    # List of URLs to scrape for plant disease information
    sources = [
        "https://www.planetnatural.com/pest-problem-solver/plant-disease/",
        "https://www.almanac.com/pest-disease",
        "https://extension.umn.edu/plant-diseases/diagnosing-plant-diseases",
        "https://www.missouribotanicalgarden.org/gardens-gardening/your-garden/help-for-the-home-gardener/advice-tips-resources/pests-and-problems.aspx",
        "https://www.gardeningknowhow.com/plant-problems/disease",
        "https://www.rhs.org.uk/prevention-protection/plant-diseases",
        "https://hortnews.extension.iastate.edu/plant-diseases",
        "https://plantvillage.psu.edu/topics"
    ]
    
    specific_diseases = [
        "https://www.planetnatural.com/pest-problem-solver/plant-disease/powdery-mildew/",
        "https://www.almanac.com/pest/blight",
        "https://www.gardeningknowhow.com/plant-problems/disease/treating-leaf-spot-fungus.htm",
        "https://www.rhs.org.uk/biodiversity/apple-scab",
        "https://plantclinic.tamu.edu/factsheets/blackspot-roses/",
        "https://extension.umn.edu/plant-diseases/tomato-leaf-spot-diseases",
        "https://www.missouribotanicalgarden.org/gardens-gardening/your-garden/help-for-the-home-gardener/advice-tips-resources/pests-and-problems/diseases/rusts/rust.aspx",
        "https://hortnews.extension.iastate.edu/downy-mildew"
    ]
    
    # Combine both general sources and specific diseases
    all_sources = sources + specific_diseases
    
    # Scrape content from each source
    for url in all_sources:
        try:
            content = get_website_text_content(url)
            if content:
                # Clean and process the content
                paragraphs = content.split('\n')
                for paragraph in paragraphs:
                    if len(paragraph.strip()) > 100:  # Only include substantial paragraphs
                        disease_data.append({
                            "source": url,
                            "content": paragraph.strip()
                        })
            
            # Add a delay to be respectful to websites
            time.sleep(random.uniform(1, 3))
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
    
    # Save the data to JSON
    with open('data_for_rag.json', 'w') as f:
        json.dump(disease_data, f, indent=2)
    
    logger.info(f"Scraped {len(disease_data)} content items from {len(all_sources)} sources")
    return disease_data

if __name__ == "__main__":
    scrape_plant_disease_data()
