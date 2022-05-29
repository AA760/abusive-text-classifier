let contextMenuItem1 = {
    "id": "FalsePositive",
    "title": "Report as abusive",
    "contexts": ["selection"]
};
chrome.contextMenus.create(contextMenuItem1)

chrome.contextMenus.onClicked.addListener(function (clickData1) {
    if (clickData1.menuItemId == "FalsePositive" && clickData1.selectionText) {
        let xhr = new XMLHttpRequest();
        xhr.open('POST', 'http://127.0.0.1:5000/reportFP', true)
        xhr.setRequestHeader('Content-Type', 'text');
        xhr.setRequestHeader('Accept', '*/*');
        xhr.send(clickData1.selectionText)
    }
})


let contextMenuItem2 = {
    "id": "FalseNegative",
    "title": "Report as NOT abusive",
    "contexts": ["selection"]
};
chrome.contextMenus.create(contextMenuItem2)

chrome.contextMenus.onClicked.addListener(function (clickData2) {
    if (clickData2.menuItemId == "FalseNegative" && clickData2.selectionText) {
        let xhr = new XMLHttpRequest();
        xhr.open('POST', 'http://127.0.0.1:5000/reportFN', true)
        xhr.setRequestHeader('Content-Type', 'text');
        xhr.setRequestHeader('Accept', '*/*');
        xhr.send(clickData2.selectionText)
    }
})


