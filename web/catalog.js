export async function readJSON(path) {
    const response = await fetch(new URL(path, import.meta.url));
    if (!response.ok) throw new Error('目录加载失败');
    return response.json();
}

export function textElement(tag, className, text) {
    const element = document.createElement(tag);
    element.className = className;
    element.textContent = text;
    return element;
}
