/* Simple scraper to get letters from James Wilkinson from Founders Online */

const cheerio = require('cheerio');
const fetch = require('node-fetch');
const fs = require('fs');

let promises = ((Array(200).fill(0))
    .map((_, i) => {
        let url = `https://founders.archives.gov/?q=%20Author%3A%22Wilkinson%2C%20James%22&s=1121311111&r=${i+1}&sr=`;
        console.log('URL', url)
        return fetch(url)
            .then(res => {
                return res.text()
            })
            .then(body => {
                let $ = cheerio.load(body);
                return $('.docbody').text()
            })
            .catch(err => console.error(err))
    })
)

console.log(promises)


Promise.all(promises).then(
    promises => fs.writeFile('wilkinson_letters.txt', promises.join('\n'), console.error)
)
.catch(console.error)