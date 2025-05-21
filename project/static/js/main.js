// src/js/main.js
// 定义一个异步函数来处理封面生成
// 该函数在 DOM 内容加载完成后执行
// 该函数会添加一个点击事件监听器到生成封面按钮，当按钮被点击时，它会获取输入的描述文本
// 如果描述文本为空，则弹出提示，否则，它会发送一个 POST 请求到指定的 URL
// 请求的内容类型为 JSON，请求体中包含描述文本
// 如果请求成功，它会解析响应并更新封面预览
// 如果响应状态为成功，则更新封面预览；否则，弹出错误提示
// 移除外层的 generateCover 函数包装
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM加载完成');
    
    // 获取按钮和输入框元素
    const generateCoverBtn = document.getElementById('generate-cover-btn');
    const coverDescriptionInput = document.getElementById('cover-description');
    const coverPreview = document.getElementById('cover-preview');
    
    console.log('按钮元素:', generateCoverBtn);
    console.log('输入框元素:', coverDescriptionInput);
    
    if (!generateCoverBtn || !coverDescriptionInput) {
        console.error('未找到必要的页面元素');
        return;
    }

    // 添加点击事件监听器
    generateCoverBtn.onclick = async function(e) {
        e.preventDefault();
        console.log('按钮被点击');
        
        const description = coverDescriptionInput.value.trim();
        console.log('输入描述:', description);

        if (!description) {
            alert('请输入封面描述！');
            return;
        }

        try {
            console.log('准备发送请求...');
            const response = await fetch('http://10.160.4.190:5000/generate-cover', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({ description })
            });

            console.log('请求已发送，等待响应...');
            const data = await response.json();
            console.log('收到响应:', data);

            if (data.status === 'success') {
                coverPreview.innerHTML = `
                    <img src="${data.image}" 
                         alt="生成封面" 
                         style="max-width: 100%; border: 2px solid hotpink;">
                `;
                console.log('图片已显示');
            } else {
                alert('生成失败: ' + data.error);
            }
        } catch (error) {
            console.error('请求失败:', error);
            alert('请求失败: ' + error.message);
        }
    };

    console.log('事件监听器已添加');
});