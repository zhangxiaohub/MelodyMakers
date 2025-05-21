// src/js/main.js

async function generateCover() {
    document.querySelector('#generate-cover-btn').addEventListener('click', async () => {
        const description = document.querySelector('#cover-description').value;

        try {
            const response = await fetch('http://10.160.4.190:5000/generate-cover', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ description })
            });

            const data = await response.json();
            if (data.status === 'success') {
                document.querySelector('#cover-preview').innerHTML = `
                    <img src="${data.image}" 
                        alt="生成封面" 
                        style="max-width: 100%; border: 2px solid hotpink;">
                `;
            } else {
                alert('生成失败: ' + data.error);
            }
        } catch (error) {
            alert('请求失败: ' + error.message);
        }
    });
}