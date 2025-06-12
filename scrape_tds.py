"""
TDS Course Material Harvester
This module collects structured content from the TDS course website
for subsequent indexing and knowledge retrieval
"""
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from tqdm import tqdm
import json
import time
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tds-material-harvester")

class CourseContentHarvester:
    """Manages the extraction of course materials from the TDS website"""
    
    def __init__(self):
        self.base_url = "https://tds.s-anand.net"
        self.collected_materials = []
        
    def harvest_course_content(self):
        """Main method to extract and save course materials"""
        logger.info("🚀 Starting TDS course material collection process")
        start_time = time.time()
        
        with sync_playwright() as browser_factory:
            # Initialize browser session
            browser_instance = browser_factory.chromium.launch(
                headless=False,
                slow_mo=50  # Slow down operations for stability
            )
            
            browser_page = browser_instance.new_page()
            browser_page.set_viewport_size({"width": 1280, "height": 800})
            
            # Access course homepage
            logger.info("🌐 Accessing TDS course homepage")
            browser_page.goto(f"{self.base_url}/#/")
            browser_page.wait_for_load_state("networkidle")
            time.sleep(2)  # Additional wait for stability
            
            # Expand navigation structure
            self._expand_course_navigation(browser_page)
            
            # Extract all material links
            material_links = self._extract_material_links(browser_page)
            logger.info(f"📚 Found {len(material_links)} course materials")
            
            # Process each material
            for index, material in enumerate(tqdm(material_links, desc="Harvesting course materials")):
                try:
                    material_data = self._process_material(browser_page, material, index)
                    if material_data:
                        self.collected_materials.append(material_data)
                except Exception as error:
                    logger.error(f"❌ Error processing material '{material['title']}': {str(error)}")
            
            browser_instance.close()
        
        # Save collected materials
        self._save_materials()
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Content collection completed in {elapsed_time:.1f} seconds")
        
    def _expand_course_navigation(self, page):
        """Expand all navigation folders to reveal content links"""
        logger.info("📂 Expanding course navigation structure")
        
        # Wait for sidebar to be fully loaded
        page.wait_for_selector("aside.sidebar", state="visible")
        time.sleep(1)
        
        try:
            # Find and click all expandable elements
            for attempt in range(3):  # Multiple attempts to ensure complete expansion
                # Find all closed folders and files
                expandable_elements = page.query_selector_all(
                    "li.folder.level-1:not(.open), li.file > div:not(.open)"
                )
                
                if not expandable_elements:
                    break
                    
                logger.info(f"📭 Found {len(expandable_elements)} expandable items (attempt {attempt+1})")
                for element in expandable_elements:
                    try:
                        element.click()
                        time.sleep(0.2)  # Brief pause between clicks
                    except Exception:
                        pass  # Skip elements that can't be clicked
                        
                time.sleep(0.5)  # Wait for DOM updates
        except Exception as error:
            logger.warning(f"⚠️ Navigation expansion issue: {str(error)}")
    
    def _extract_material_links(self, page) -> List[Dict[str, str]]:
        """Extract all material links from the expanded navigation"""
        link_elements = page.query_selector_all("li.file a")
        
        return [
            {
                "title": self._clean_text(element.inner_text()),
                "href": element.get_attribute("href") or ""
            }
            for element in link_elements
            if element.get_attribute("href")
        ]
    
    def _process_material(self, page, material: Dict[str, str], index: int) -> Optional[Dict[str, str]]:
        """Extract content from a course material page"""
        material_url = f"{self.base_url}/{material['href']}"
        
        # Navigate to material page
        page.goto(material_url)
        
        try:
            # Wait for content to load
            page.wait_for_selector(".content", timeout=8000)
            time.sleep(0.5)  # Allow time for content to fully render
            
            # Extract content using page evaluation for best results
            material_content = page.locator(".content").inner_text()
            
            # Display a preview
            content_preview = material_content[:150] + "..." if len(material_content) > 150 else material_content
            logger.debug(f"📝 Material {index+1}: {material['title']}")
            
            return {
                "title": material["title"],
                "content": material_content.strip(),
                "url": material_url,
                "collected_at": datetime.now().isoformat()
            }
            
        except Exception as error:
            logger.error(f"❌ Failed to process '{material['title']}': {str(error)}")
            return None
    
    def _save_materials(self):
        """Save collected materials to JSON file"""
        # Ensure output directory exists
        os.makedirs("data", exist_ok=True)
        output_path = "data/tds_content.json"
        
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(self.collected_materials, output_file, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Saved {len(self.collected_materials)} materials to {output_path}")
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean and normalize text content"""
        return " ".join(text.split())

def harvest_tds_site():
    """Entry point function to start the harvesting process"""
    harvester = CourseContentHarvester()
    harvester.harvest_course_content()

if __name__ == "__main__":
    harvest_tds_site()