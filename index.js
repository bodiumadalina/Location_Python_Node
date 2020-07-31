const express = require('express')
const { spawn } = require('child_process')
const app = express()
const port = 3000





app.get('/', (req, res) => {
  let dataToSend
  let largeDataSet = []
  let location=[]
  // spawn new child process to call the python script
  const python = spawn('python', ['predict.py', "DSC_0395.JPG"])
  //location=python('data', function (data)

  // collect data from script
  python.stdout.on('data', function (data) {
    console.log('Pipe data from python script ...')
    //dataToSend =  data;
    location.push(data)
  })

  // in close event we are sure that stream is from child process is closed
  python.on('close', (code) => {
    console.log(`child process close all stdio with code ${code}`)
    // send data to browser
    res.send(location.join(''))
  })
})

app.listen(port, () => {
  console.log(`App listening on port ${port}!`)
})
