import { readJSON, textElement } from './catalog.js';

const audio = document.querySelector('#audioPlayer');
const status = document.querySelector('#nowPlaying');
const list = document.querySelector('#playlist');
const gaps = document.querySelector('#unavailable');
const previous = document.querySelector('#previous');
const next = document.querySelector('#next');
const bookId = new URLSearchParams(location.search).get('bookId');

let tracks = [];
let current = 0;
let restoreAt = 0;
let lastWrite = 0;
const progressKey = 'tingshu:progress:' + bookId;

function savedProgress() {
    try {
        return JSON.parse(localStorage.getItem(progressKey));
    } catch {
        return null;
    }
}

function saveProgress(force = false) {
    if (!tracks[current] || audio.readyState < 1 || restoreAt > 0) return;
    const now = Date.now();
    if (!force && now - lastWrite < 4000) return;
    lastWrite = now;
    try {
        localStorage.setItem(progressKey, JSON.stringify({
            src: tracks[current].src,
            time: audio.currentTime,
        }));
    } catch {
        // Playback is still available when browser storage is disabled.
    }
}

function selectTrack(index, { play = false, time = 0 } = {}) {
    const track = tracks[index];
    if (!track) return;
    if (play && index === current && audio.src === new URL(track.src, document.baseURI).href && !audio.error) {
        playAudio();
        return;
    }
    current = index;
    restoreAt = Number.isFinite(time) && time > 0 ? time : 0;
    audio.src = track.src;
    status.textContent = '当前章节：' + track.title;
    previous.disabled = index === 0;
    next.disabled = index === tracks.length - 1;
    for (const [i, button] of Array.from(list.querySelectorAll('button')).entries()) {
        if (i === current) button.setAttribute('aria-current', 'true');
        else button.removeAttribute('aria-current');
    }
    if (play) playAudio();
}

function playAudio() {
    audio.play().catch(error => {
        if (error.name !== 'AbortError') status.textContent = '请按播放器的播放键继续';
    });
}

audio.addEventListener('loadedmetadata', () => {
    if (restoreAt && restoreAt < audio.duration - 1) audio.currentTime = restoreAt;
    restoreAt = 0;
    saveProgress(true);
});
audio.addEventListener('playing', () => {
    if (tracks[current]) status.textContent = '当前章节：' + tracks[current].title;
});
audio.addEventListener('timeupdate', () => saveProgress());
audio.addEventListener('pause', () => saveProgress(true));
audio.addEventListener('seeked', () => saveProgress(true));
audio.addEventListener('error', () => {
    status.textContent = '本章音频加载失败，可重试或切换章节。';
});
audio.addEventListener('ended', () => {
    saveProgress(true);
    if (current + 1 < tracks.length) selectTrack(current + 1, { play: true });
    else status.textContent = '已播放到最后一章。';
});
window.addEventListener('pagehide', () => saveProgress(true));
previous.addEventListener('click', () => selectTrack(current - 1, { play: true }));
next.addEventListener('click', () => selectTrack(current + 1, { play: true }));

try {
    const book = (await readJSON('catalog.json')).find(entry => entry.id === bookId);
    if (!book) {
        status.textContent = '请先从书架选择一本书。';
    } else {
        document.querySelector('#bookTitle').textContent = book.title;
        if (!book.manifest) {
            status.textContent = '本书音频待上架。';
        } else {
            const manifest = await readJSON(book.manifest);
            tracks = manifest.audios;
            if (!tracks.length) {
                status.textContent = '本书音频待上架。';
            } else {
                for (const [index, track] of tracks.entries()) {
                    const item = document.createElement('li');
                    const button = textElement('button', 'playlist-item', track.title);
                    button.addEventListener('click', () => selectTrack(index, { play: true }));
                    item.append(button);
                    list.append(item);
                }
                const unavailable = manifest.unavailable ?? [];
                if (unavailable.length) {
                    gaps.hidden = false;
                    const items = gaps.querySelector('ul');
                    for (const title of unavailable) items.append(textElement('li', '', title));
                }
                document.querySelector('#synthetic-notice').hidden = false;
                document.querySelector('#player-controls').hidden = false;
                list.hidden = false;
                const saved = savedProgress();
                const savedIndex = saved ? tracks.findIndex(track => track.src === saved.src) : -1;
                selectTrack(Math.max(savedIndex, 0), { time: savedIndex >= 0 ? saved.time : 0 });
            }
        }
    }
} catch {
    status.textContent = '音频清单加载失败，请刷新页面重试。';
}
