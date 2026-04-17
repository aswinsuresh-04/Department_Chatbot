import json
import os
import time
from urllib.parse import urljoin, urlparse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

BASE_URL = "https://cs.cusat.ac.in"
MAX_PAGES = 100

# Pages to skip (PDFs, external, images)
SKIP_EXTENSIONS = ('.pdf', '.jpg', '.jpeg', '.png', '.zip', '.doc', '.docx', '.svg')
SKIP_DOMAINS = ['fonts.googleapis.com', 'cusat.ac.in', 'cittic.cusat.ac.in',
                'icaise.cusat.ac.in', 'malayalamwordnet.cusat.ac.in', 'google.com']

def get_driver():
    options = Options()
    options.add_argument('--headless')           # Run in background
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--log-level=3')        # Suppress console noise
    options.add_experimental_option('excludeSwitches', ['enable-logging'])

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver


def is_valid_url(url):
    parsed = urlparse(url)
    # Must be same domain
    if parsed.netloc and parsed.netloc != urlparse(BASE_URL).netloc:
        return False
    # Skip file types
    if any(parsed.path.lower().endswith(ext) for ext in SKIP_EXTENSIONS):
        return False
    # Skip anchors only
    if url.startswith('#'):
        return False
    return True


def extract_page_data(driver, url):
    """Extract structured data from a page."""
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    # Remove noise
    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()

    # --- Extract structured content ---

    # Title
    title = soup.title.string.strip() if soup.title else ''

    # Navigation links
    nav_links = []
    for a in soup.select('nav a, .stellarnav a'):
        text = a.get_text(strip=True)
        href = a.get('href', '')
        if text and href:
            nav_links.append({'text': text, 'href': href})

    # News & Events cards
    news_items = []
    for card in soup.select('.news-card-link'):
        heading = card.select_one('.blue-bold-heading')
        img = card.select_one('img.card-img-top')
        href = card.get('href', '')
        news_items.append({
            'title': heading.get_text(strip=True) if heading else '',
            'link': href,
            'image': img.get('src', '') if img else ''
        })

    # Ticker / announcements
    announcements = []
    for a in soup.select('.marquee a, .ticker a'):
        text = a.get_text(strip=True)
        href = a.get('href', '')
        if text:
            announcements.append({'text': text, 'link': href})

    # Courses
    courses = []
    for card in soup.select('.courses-card'):
        heading = card.select_one('.blue-bold-heading')
        features = [p.get_text(strip=True) for p in card.select('.courses-feature')]
        link = card.select_one('a.courses-button')
        courses.append({
            'name': heading.get_text(strip=True) if heading else '',
            'features': features,
            'link': link.get('href', '') if link else ''
        })

    # Vision & Mission
    vision = soup.select_one('.vision-card p')
    mission_items = [li.get_text(strip=True) for li in soup.select('.mission-card li')]

    # Contact info
    contacts = []
    for ct in soup.select('.contact-text, .footer-text'):
        text = ct.get_text(strip=True)
        if text:
            contacts.append(text)

    # People / Faculty (for people.php)
    people = []
    for person in soup.select('.faculty-card, .people-card, .card'):
        name = person.select_one('h5, h4, .card-title')
        designation = person.select_one('.designation, .card-text, p')
        if name:
            people.append({
                'name': name.get_text(strip=True),
                'designation': designation.get_text(strip=True) if designation else ''
            })

    # General main text (fallback for unique pages)
    main = soup.select_one('main, .main-content, #main, article')
    if not main:
        main = soup.select_one('body')
    main_text = main.get_text(separator='\n', strip=True) if main else ''
    # Trim excessive whitespace
    main_text = '\n'.join(line for line in main_text.splitlines() if line.strip())

    # Collect internal links from this page
    internal_links = []
    for a in soup.find_all('a', href=True):
        href = a['href'].strip()
        full_url = urljoin(url, href)
        if is_valid_url(full_url) and urlparse(full_url).netloc == urlparse(BASE_URL).netloc:
            internal_links.append(full_url)

    return {
        'url': url,
        'title': title,
        'nav_links': nav_links,
        'announcements': announcements,
        'news_events': news_items,
        'courses': courses,
        'vision': vision.get_text(strip=True) if vision else '',
        'mission': mission_items,
        'contacts': contacts,
        'people': people,
        'main_text': main_text[:5000],  # Cap to avoid huge blobs
        '_internal_links': internal_links  # Used for crawling, stripped later
    }, internal_links


def run_scraper():
    print(f"Starting Selenium scrape of {BASE_URL}")
    driver = get_driver()

    visited = set()
    results = []
    queue = [BASE_URL]

    try:
        while queue and len(visited) < MAX_PAGES:
            url = queue.pop(0)

            # Normalize URL
            url = url.rstrip('/')
            if url in visited:
                continue
            visited.add(url)

            print(f"[{len(visited):03d}] Scraping: {url}")

            try:
                driver.get(url)
                # Wait for body to load
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, 'body'))
                )
                time.sleep(1)  # Allow JS to render

                page_data, new_links = extract_page_data(driver, url)

                # Remove internal crawl helper key before saving
                page_data.pop('_internal_links', None)
                results.append(page_data)

                # Add new unvisited links to queue
                for link in new_links:
                    link = link.rstrip('/')
                    if link not in visited and link not in queue:
                        queue.append(link)

            except Exception as e:
                print(f"  ⚠️  Failed: {url} — {e}")
                continue

    finally:
        driver.quit()

    # Save results
    output_file = 'cusat_data.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Done. Scraped {len(results)} pages.")
    print(f"📁 Saved to: {os.path.abspath(output_file)}")


if __name__ == "__main__":
    run_scraper()