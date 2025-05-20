const video = document.getElementById('video');
const resultado = document.getElementById('resultado');

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video.srcObject = stream;
  });

setInterval(() => {
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  const image = canvas.toDataURL('image/jpeg');

  fetch('/reconhecer', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image })
  })
    .then(response => response.json())
    .then(data => {
      resultado.innerText = data.length > 0
        ? data.map(p => `${p.nome} (${p.probabilidade})`).join(", ")
        : "Nenhuma face detectada.";
    });
}, 2000);
