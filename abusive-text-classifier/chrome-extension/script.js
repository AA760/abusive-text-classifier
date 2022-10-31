function censorText(element) {
	if (element.hasChildNodes()) {
        element.childNodes.forEach(censorText)
    } else if (element.nodeType === Text.TEXT_NODE) {
        if (element.textContent && element.textContent !== null && element.textConent !== "/n" && element.textContent.match('[a-zA-Z]')
            && element.textContent.match('^((?!Reply|Share|Save|Report|Follow|Edited).)*$') && element.textContent.match('^((?!\d+\s[a-z]+.\sago).)*$')) {
            let xhr = new XMLHttpRequest();
            xhr.open('POST', 'http://127.0.0.1:5000/', true)
            xhr.setRequestHeader('Content-Type', 'text');
            xhr.setRequestHeader('Accept', '*/*');
            xhr.send(element.textContent)
            xhr.onreadystatechange = function () {
                if (xhr.readyState === xhr.DONE) {
                    if (xhr.status === 200) {
                        if (xhr.responseText === "1") {
                            element.parentElement.style.color = 'black'
                            element.parentElement.style.backgroundColor = 'black'
                        }
                    }
                }
            };
        }
    }
}

document.onreadystatechange = function () {
    if (document.readyState === 'complete') {
        censorText(document.body)
    }
}

let observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
        if (!mutation.addedNodes) return

        for (let i = 0; i < mutation.addedNodes.length; i++) {
            let node = mutation.addedNodes[i]
            censorText(node)
        }
    })
})

observer.observe(document.body, {
    childList: true
    , subtree: true
    , attributes: false
    , characterData: false
})

