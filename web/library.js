import { readJSON, textElement } from './catalog.js';

const grid = document.querySelector('.books-grid');
const search = document.querySelector('#book-search');
const status = document.querySelector('#catalog-status');

function makeCard(book) {
    const card = document.createElement(book.manifest ? 'a' : 'article');
    card.className = 'book-card';
    if (book.manifest) card.href = 'book.html?' + new URLSearchParams({ bookId: book.id });

    const cover = document.createElement('img');
    cover.src = book.cover;
    cover.alt = '';
    cover.className = 'book-cover';
    cover.width = 120;
    cover.height = 160;
    cover.loading = 'lazy';
    card.append(
        cover,
        textElement('h3', 'book-title', book.title),
        textElement('p', 'book-author', book.author),
        textElement('p', 'reading-status', book.wordCount),
        textElement('p', 'book-status', book.manifest ? '可收听 · AI 合成朗读' : '音频待上架'),
    );
    return card;
}

try {
    const books = await readJSON('catalog.json');
    const entries = books.map(book => ({
        searchText: (book.title + ' ' + book.author).normalize('NFKC').toLowerCase(),
        card: makeCard(book),
    }));
    grid.append(...entries.map(entry => entry.card));

    function filter() {
        const terms = search.value.normalize('NFKC').toLowerCase().trim().split(/\s+/);
        let shown = 0;
        for (const entry of entries) {
            const matches = terms.every(term => entry.searchText.includes(term));
            entry.card.hidden = !matches;
            if (matches) shown += 1;
        }
        status.textContent = shown ? '' : '没有找到相关书籍，请换个关键词。';
    }

    search.disabled = false;
    search.addEventListener('input', filter);
    filter();
} catch {
    status.textContent = '书架加载失败，请刷新页面重试。';
}
