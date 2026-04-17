// scraper.js
const puppeteer = require('puppeteer-extra');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
const fs = require('fs-extra');
const path = require('path');
const axios = require('axios');
const { URL } = require('url');
const TurndownService = require('turndown');

puppeteer.use(StealthPlugin());

const BASE_URL = process.argv[2] || 'https://cs.cusat.ac.in';
const MAX_PAGES = 200;
const OUTPUT_DIR = 'scraped_content';
const PDF_DIR = path.join(OUTPUT_DIR, 'pdfs');
const MD_DIR = path.join(OUTPUT_DIR, 'pages');

const SKIP_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.zip', '.gif', '.mp4', '.mp3', '.svg', '.ico', '.webp'];

const turndown = new TurndownService({
    headingStyle: 'atx',
    bulletListMarker: '-',
    codeBlockStyle: 'fenced'
});

// Remove noisy elements from markdown conversion
turndown.remove(['script', 'style', 'noscript', 'iframe', 'meta', 'link']);

// Remove links but keep their text
turndown.addRule('cleanLinks', {
    filter: 'a',
    replacement: (content) => content.trim() ? content.trim() : ''
});

// Remove images entirely
turndown.addRule('removeImages', {
    filter: 'img',
    replacement: () => ''
});

function normalizeUrl(url) {
    try {
        const u = new URL(url);
        u.hash = '';
        return u.href.replace(/\/$/, '');
    } catch {
        return null;
    }
}

function isSameDomain(url) {
    try {
        return new URL(url).hostname === new URL(BASE_URL).hostname;
    } catch {
        return false;
    }
}

function isPdf(url) {
    try {
        return new URL(url).pathname.toLowerCase().endsWith('.pdf');
    } catch {
        return false;
    }
}

function shouldSkip(url) {
    try {
        const pathname = new URL(url).pathname.toLowerCase();
        return SKIP_EXTENSIONS.some(ext => pathname.endsWith(ext));
    } catch {
        return true;
    }
}

function slugify(str) {
    return str
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '_')
        .replace(/^_+|_+$/g, '')
        .slice(0, 80) || 'page';
}

function cleanText(text) {
    return text
        .replace(/\t/g, ' ')
        .replace(/ {2,}/g, ' ')
        .replace(/\n{3,}/g, '\n\n')
        .trim();
}

async function downloadPdf(url, cookies) {
    try {
        const filename = decodeURIComponent(path.basename(new URL(url).pathname));
        const filepath = path.join(PDF_DIR, filename);

        if (await fs.pathExists(filepath)) {
            const stat = await fs.stat(filepath);
            if (stat.size > 1000) {
                console.log(`  📄 Already downloaded: ${filename}`);
                return filename;
            }
        }

        const cookieHeader = cookies.map(c => `${c.name}=${c.value}`).join('; ');

        const https = require('https');
        const response = await axios.get(url, {
            responseType: 'arraybuffer',
            timeout: 30000,
            // ✅ Fix 1: Disable SSL certificate verification
            httpsAgent: new https.Agent({ rejectUnauthorized: false }),
            headers: {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Cookie': cookieHeader,
                'Referer': BASE_URL,
                'Accept': 'application/pdf,*/*'
            }
        });

        const buffer = Buffer.from(response.data);

        // ✅ Fix 2: Validate it's actually a PDF (starts with %PDF)
        if (buffer.length < 100) {
            console.log(`  ⚠️  Skipped (too small): ${filename}`);
            return null;
        }

        const header = buffer.slice(0, 5).toString('ascii');
        if (!header.startsWith('%PDF')) {
            console.log(`  ⚠️  Skipped (not a valid PDF): ${filename}`);
            return null;
        }

        await fs.writeFile(filepath, buffer);
        console.log(`  📥 Downloaded: ${filename} (${(buffer.length / 1024).toFixed(1)} KB)`);
        return filename;

    } catch (err) {
        console.log(`  ⚠️  PDF failed: ${path.basename(url)} — ${err.message}`);
        return null;
    }
}

async function extractCleanContent(page, url) {
    // Remove all noise elements before extracting
    await page.evaluate(() => {
        const noisy = [
            'script', 'style', 'noscript', 'iframe',
            'header', 'footer', 'nav', '.stellarnav',
            '.footer', '.header', '#header-sticky',
            '.header-top-bar', '.back-to-top-button',
            '#back-to-top-button', '.cookie-banner',
            '.social-icons', '.qr-code', '.footer-disclaimer-section'
        ];
        noisy.forEach(sel => {
            document.querySelectorAll(sel).forEach(el => el.remove());
        });
    });

    const data = await page.evaluate((pageUrl) => {
        const title = document.title?.trim() || '';

        // Page-specific extractors
        const getSection = (selector) => {
            const el = document.querySelector(selector);
            return el ? el.innerText.replace(/\s+/g, ' ').trim() : '';
        };

        const getAllText = (selector) =>
            [...document.querySelectorAll(selector)]
                .map(el => el.innerText.trim())
                .filter(Boolean);

        // --- Structured extraction ---

        // Announcements / ticker
        const announcements = [...document.querySelectorAll('.marquee a, .ticker a, .marquee-content a')]
            .map(a => a.innerText.trim())
            .filter(Boolean);

        // News & Events
        const newsEvents = [...document.querySelectorAll('.news-card-link, .card')]
            .map(card => {
                const heading = card.querySelector('.blue-bold-heading, h5, h4');
                return heading ? heading.innerText.trim() : '';
            })
            .filter(Boolean);

        // Courses
        const courses = [...document.querySelectorAll('.courses-card')]
            .map(card => {
                const name = card.querySelector('.blue-bold-heading')?.innerText.trim() || '';
                const features = [...card.querySelectorAll('.courses-feature')]
                    .map(f => f.innerText.trim()).join(', ');
                return name ? `${name}${features ? ' — ' + features : ''}` : '';
            })
            .filter(Boolean);

        // Vision
        const vision = getSection('.vision-card p, .vision-card');

        // Mission
        const mission = getAllText('.mission-card li');

        // People / Faculty
        const people = [...document.querySelectorAll('.faculty-card, .people-card, .staff-card, .faculty')]
            .map(card => {
                const name = card.querySelector('h4, h5, .name, strong')?.innerText.trim() || '';
                const desig = card.querySelector('.designation, .role, small, span')?.innerText.trim() || '';
                const email = card.querySelector('a[href^="mailto"]')?.href?.replace('mailto:', '') || '';
                const research = card.querySelector('.research-area, .specialization')?.innerText.trim() || '';
                return name ? { name, designation: desig, email, research } : null;
            })
            .filter(Boolean);

        // Contact
        const emails = [...new Set([...document.querySelectorAll('a[href^="mailto"]')]
            .map(a => a.href.replace('mailto:', '')))];
        const phones = [...new Set([...document.querySelectorAll('a[href^="tel"]')]
            .map(a => a.href.replace('tel:', '')))];
        const address = getSection('.footer-address, address, .address');

        // All headings for structure
        const headings = [...document.querySelectorAll('h1, h2, h3')]
            .map(h => h.innerText.trim())
            .filter(Boolean);

        // Main content HTML (cleaned, for markdown conversion)
        const mainEl = document.querySelector(
            'main, .main-content, #main, article, .container, [role="main"]'
        ) || document.body;

        const mainHtml = mainEl ? mainEl.innerHTML : '';

        // All PDF links
        const pdfLinks = [...document.querySelectorAll('a[href]')]
            .map(a => a.href)
            .filter(href => href && href.toLowerCase().endsWith('.pdf'));

        // All internal links
        const internalLinks = [...document.querySelectorAll('a[href]')]
            .map(a => a.href)
            .filter(href => href && !href.startsWith('mailto:') && !href.startsWith('tel:') && !href.startsWith('javascript:'));

        return {
            url: pageUrl,
            title,
            headings,
            announcements,
            newsEvents,
            courses,
            vision,
            mission,
            people,
            contact: { emails, phones, address },
            mainHtml,
            pdfLinks,
            internalLinks
        };
    }, url);

    return data;
}

function buildMarkdown(data, downloadedPdfs) {
    const lines = [];
    const hostname = new URL(BASE_URL).hostname;
    const pageSlug = slugify(data.title || data.url);

    lines.push(`# ${data.title || 'Untitled Page'}`);
    lines.push(`**Source:** ${data.url}`);
    lines.push(`**Website:** ${hostname}`);
    lines.push('');

    // Convert main HTML to markdown
    let mainMd = '';
    try {
        mainMd = turndown.turndown(data.mainHtml || '');
        // Clean up the markdown
        mainMd = mainMd
            .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')   // Remove links, keep text
            .replace(/!\[[^\]]*\]\([^)]*\)/g, '')        // Remove image syntax
            .replace(/\n{3,}/g, '\n\n')                  // Max 2 blank lines
            .replace(/^\s*[-*]\s*$/gm, '')               // Remove empty bullets
            .trim();
    } catch (e) {
        mainMd = '';
    }

    // --- Structured sections ---

    if (data.announcements?.length) {
        lines.push('## Announcements');
        data.announcements.forEach(a => lines.push(`- ${a}`));
        lines.push('');
    }

    if (data.newsEvents?.length) {
        lines.push('## News & Events');
        [...new Set(data.newsEvents)].forEach(n => lines.push(`- ${n}`));
        lines.push('');
    }

    if (data.courses?.length) {
        lines.push('## Courses Offered');
        data.courses.forEach(c => lines.push(`- ${c}`));
        lines.push('');
    }

    if (data.vision) {
        lines.push('## Vision');
        lines.push(data.vision);
        lines.push('');
    }

    if (data.mission?.length) {
        lines.push('## Mission');
        data.mission.forEach(m => lines.push(`- ${m}`));
        lines.push('');
    }

    if (data.people?.length) {
        lines.push('## Faculty / People');
        data.people.forEach(p => {
            lines.push(`### ${p.name}`);
            if (p.designation) lines.push(`**Designation:** ${p.designation}`);
            if (p.email) lines.push(`**Email:** ${p.email}`);
            if (p.research) lines.push(`**Research Area:** ${p.research}`);
            lines.push('');
        });
    }

    // Main content (deduplicated from structured above)
    if (mainMd) {
        lines.push('## Page Content');
        lines.push(mainMd);
        lines.push('');
    }

    // Contact
    const { emails, phones, address } = data.contact || {};
    if (emails?.length || phones?.length || address) {
        lines.push('## Contact Information');
        if (address) lines.push(`**Address:** ${address}`);
        if (emails?.length) lines.push(`**Email:** ${emails.join(', ')}`);
        if (phones?.length) lines.push(`**Phone:** ${phones.join(', ')}`);
        lines.push('');
    }

    // Downloaded PDFs reference
    if (downloadedPdfs?.length) {
        lines.push('## Related Documents (PDFs)');
        downloadedPdfs.forEach(f => lines.push(`- ${f}`));
        lines.push('');
    }

    return cleanText(lines.join('\n'));
}

async function scrape() {
    await fs.ensureDir(PDF_DIR);
    await fs.ensureDir(MD_DIR);

    console.log(`\n🚀 Scraping: ${BASE_URL}\n`);

    const browser = await puppeteer.launch({
        headless: false,
        args: [
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-blink-features=AutomationControlled',
            '--window-size=1366,768',
            '--disable-infobars',
        ],
        defaultViewport: null,
        ignoreHTTPSErrors: true,
    });

    const page = await browser.newPage();

    await page.setUserAgent(
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
    );

    await page.setExtraHTTPHeaders({
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    });

    // Block media to speed up
    await page.setRequestInterception(true);
    page.on('request', req => {
        if (['image', 'font', 'media'].includes(req.resourceType())) {
            req.abort();
        } else {
            req.continue();
        }
    });

    const visited = new Set();
    const pdfQueue = new Set();
    const queue = [BASE_URL];
    const allMarkdowns = [];
    const index = [];

    while (queue.length > 0 && visited.size < MAX_PAGES) {
        const url = queue.shift();
        const normalized = normalizeUrl(url);

        if (!normalized || visited.has(normalized)) continue;
        if (!isSameDomain(normalized)) continue;
        if (shouldSkip(normalized) || isPdf(normalized)) continue;

        visited.add(normalized);
        console.log(`[${String(visited.size).padStart(3, '0')}] ${normalized}`);

        try {
            const res = await page.goto(normalized, {
                waitUntil: 'domcontentloaded',
                timeout: 20000
            });

            if (res?.status() >= 400) {
                console.log(`  ⛔ HTTP ${res.status()}`);
                continue;
            }

            await new Promise(r => setTimeout(r, 1500));

            // Check for WAF block
            const bodyText = await page.evaluate(() => document.body?.innerText || '');
            if (/Web Page Blocked|Access Denied|403 Forbidden|captcha/i.test(bodyText)) {
                console.log(`  ⛔ Blocked`);
                continue;
            }

            const data = await extractCleanContent(page, normalized);

            // Collect PDFs
            for (const pdfUrl of data.pdfLinks || []) {
                const fullPdf = normalizeUrl(pdfUrl);
                if (fullPdf) pdfQueue.add(fullPdf);
            }

            // Queue internal links
            for (const link of data.internalLinks || []) {
                const norm = normalizeUrl(link);
                if (norm && !visited.has(norm) && !queue.includes(norm) && isSameDomain(norm)) {
                    queue.push(norm);
                }
            }

            // Get current cookies for PDF downloads
            const cookies = await page.cookies();

            // Download PDFs found on this page
            const downloadedPdfs = [];
            for (const pdfUrl of data.pdfLinks || []) {
                const filename = await downloadPdf(pdfUrl, cookies);
                if (filename) downloadedPdfs.push(filename);
            }

            // Build markdown
            const markdown = buildMarkdown(data, downloadedPdfs);
            const slug = slugify(data.title || normalized);
            const filename = `${String(visited.size).padStart(3, '0')}_${slug}.md`;
            const filepath = path.join(MD_DIR, filename);

            await fs.writeFile(filepath, markdown, 'utf-8');
            allMarkdowns.push(markdown);
            index.push({ file: filename, url: normalized, title: data.title });

            console.log(`  ✅ "${data.title}" → ${filename}`);

        } catch (err) {
            console.log(`  ⚠️  ${err.message}`);
        }

        // Human-like delay
        await new Promise(r => setTimeout(r, 800 + Math.random() * 1000));
    }

    // Download any remaining PDFs found across all pages
    console.log(`\n📥 Downloading ${pdfQueue.size} PDFs...`);
    const cookies = await page.cookies();
    for (const pdfUrl of pdfQueue) {
        await downloadPdf(pdfUrl, cookies);
        await new Promise(r => setTimeout(r, 300));
    }

    await browser.close();

    // --- Write combined master file ---
    const masterContent = allMarkdowns.join('\n\n---\n\n');
    await fs.writeFile(
        path.join(OUTPUT_DIR, 'all_content.md'),
        masterContent,
        'utf-8'
    );

    // Write index
    await fs.writeFile(
        path.join(OUTPUT_DIR, 'index.json'),
        JSON.stringify(index, null, 2),
        'utf-8'
    );

    console.log(`
╔══════════════════════════════════════════════════╗
║  ✅  Scraping Complete                           ║
╠══════════════════════════════════════════════════╣
║  Pages scraped  : ${String(index.length).padEnd(29)}║
║  Output folder  : ${OUTPUT_DIR.padEnd(29)}║
║  Individual MDs : ${OUTPUT_DIR}/pages/           ║
║  Downloaded PDFs: ${OUTPUT_DIR}/pdfs/            ║
║  Master file    : ${OUTPUT_DIR}/all_content.md   ║
╚══════════════════════════════════════════════════╝
`);
}

scrape().catch(console.error);