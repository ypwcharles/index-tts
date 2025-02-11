<div align="center">
<img src='assets/index_icon.png' width="250"/>
</div>


<h2><center>IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System</h2>

<p align="center">
<a href='https://arxiv.org/abs/2502.05512'><img src='https://img.shields.io/badge/ArXiv--red'></a>

## 👉🏻 IndexTTS 👈🏻

[[Paper]](https://arxiv.org/abs/2502.05512)  [[Demos]](https://index-tts.github.io)  

**IndexTTS** is a GPT-style text-to-speech (TTS) model mainly based on XTTS and Tortoise. It is capable of correcting the pronunciation of Chinese characters using pinyin and controlling pauses at any position through punctuation marks. We enhanced multiple modules of the system, including the improvement of speaker condition feature representation, and the integration of BigVGAN2 to optimize audio quality. Trained on tens of thousands of hours of data, our system achieves state-of-the-art performance, outperforming current popular TTS systems such as XTTS, CosyVoice2, Fish-Speech, and F5-TTS.
<span style="font-size:16px;">  
Experience **IndexTTS**: Please contact <u>xuanwu@bilibili.com</u> for more detailed information. </span>


## 🖥️ Method

The overview of IndexTTS is shown as follows.

<picture>
  <img src="assets/indexTTS.png"  width="800"/>
</picture>

The main improvements and contributions are summarized as follows:

 - In Chinese scenarios, we have introduced a character-pinyin hybrid modeling approach. This allows for quick correction of mispronounced characters.
 - **IndexTTS** incorporate a conformer conditioning encoder and a BigVGAN2-based speechcode decoder. This improves training stability, voice timbre similarity, and sound quality.
 - We release all test sets here, including those for polysyllabic words, subjective and objective test sets.

## 📣 Updates

- `2025/02/12` 🔥🔥We submitted our paper on arXiv, and released our demos and test sets.
- [WIP] We plan to release the model parameters and code in a few weeks.


## 📑 Evaluation

**Word Error Rate (WER) and Speaker Similarity (SS) Results for IndexTTS and Baseline Models**

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-7btt{border-color:inherit;font-weight:bold;text-align:center;vertical-align:top}
</style>
<table class="tg"><thead>
  <tr>
    <th class="tg-7btt" rowspan="2">Model</th>
    <th class="tg-7btt" colspan="2">aishell1_test</th>
    <th class="tg-7btt" colspan="2">commonvoice_20_test_zh</th>
    <th class="tg-7btt" colspan="2">commonvoice_20_test_en</th>
    <th class="tg-7btt" colspan="2">librispeech_test_clean</th>
    <th class="tg-7btt" colspan="2">avg</th>
  </tr>
  <tr>
    <th class="tg-7btt">CER(%)↓</th>
    <th class="tg-7btt">SS↑</th>
    <th class="tg-7btt">CER(%)↓</th>
    <th class="tg-7btt">SS↑</th>
    <th class="tg-7btt">WER(%)↓</th>
    <th class="tg-7btt">SS↑</th>
    <th class="tg-7btt">WER(%)↓</th>
    <th class="tg-7btt">SS↑</th>
    <th class="tg-7btt">CER(%)↓</th>
    <th class="tg-7btt">SS↑</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-7btt">Human</td>
    <td class="tg-c3ow">2.0 </td>
    <td class="tg-c3ow">0.846</td>
    <td class="tg-c3ow">9.5 </td>
    <td class="tg-c3ow">0.809</td>
    <td class="tg-c3ow">10.0 </td>
    <td class="tg-c3ow">0.820</td>
    <td class="tg-c3ow">2.4 </td>
    <td class="tg-c3ow">0.858</td>
    <td class="tg-c3ow">5.1 </td>
    <td class="tg-c3ow">0.836</td>
  </tr>
  <tr>
    <td class="tg-7btt">CosyVoice 2</td>
    <td class="tg-c3ow">1.8 </td>
    <td class="tg-7btt">0.796</td>
    <td class="tg-c3ow">9.1 </td>
    <td class="tg-c3ow">0.743</td>
    <td class="tg-c3ow">7.3 </td>
    <td class="tg-c3ow">0.742</td>
    <td class="tg-c3ow">4.9 </td>
    <td class="tg-7btt">0.837</td>
    <td class="tg-c3ow">5.9 </td>
    <td class="tg-7btt">0.788</td>
  </tr>
  <tr>
    <td class="tg-7btt">F5TTS</td>
    <td class="tg-c3ow">3.9 </td>
    <td class="tg-c3ow">0.743</td>
    <td class="tg-c3ow">11.7 </td>
    <td class="tg-7btt">0.747</td>
    <td class="tg-c3ow">5.4 </td>
    <td class="tg-c3ow">0.746</td>
    <td class="tg-c3ow">7.8 </td>
    <td class="tg-c3ow">0.828</td>
    <td class="tg-c3ow">8.2 </td>
    <td class="tg-c3ow">0.779</td>
  </tr>
  <tr>
    <td class="tg-7btt">Fishspeech</td>
    <td class="tg-c3ow">2.4 </td>
    <td class="tg-c3ow">0.488</td>
    <td class="tg-c3ow">11.4 </td>
    <td class="tg-c3ow">0.552</td>
    <td class="tg-c3ow">8.8 </td>
    <td class="tg-c3ow">0.622</td>
    <td class="tg-c3ow">8.0 </td>
    <td class="tg-c3ow">0.701</td>
    <td class="tg-c3ow">8.3 </td>
    <td class="tg-c3ow">0.612</td>
  </tr>
  <tr>
    <td class="tg-7btt">FireRedTTS</td>
    <td class="tg-c3ow">2.2 </td>
    <td class="tg-c3ow">0.579</td>
    <td class="tg-c3ow">11.0 </td>
    <td class="tg-c3ow">0.593</td>
    <td class="tg-c3ow">16.3 </td>
    <td class="tg-c3ow">0.587</td>
    <td class="tg-c3ow">5.7 </td>
    <td class="tg-c3ow">0.698</td>
    <td class="tg-c3ow">7.7 </td>
    <td class="tg-c3ow">0.631</td>
  </tr>
  <tr>
    <td class="tg-7btt">XTTS</td>
    <td class="tg-c3ow">3.0 </td>
    <td class="tg-c3ow">0.573</td>
    <td class="tg-c3ow">11.4 </td>
    <td class="tg-c3ow">0.586</td>
    <td class="tg-c3ow">7.1 </td>
    <td class="tg-c3ow">0.648</td>
    <td class="tg-c3ow">3.5 </td>
    <td class="tg-c3ow">0.761</td>
    <td class="tg-c3ow">6.0 </td>
    <td class="tg-c3ow">0.663</td>
  </tr>
  <tr>
    <td class="tg-7btt">IndexTTS</td>
    <td class="tg-7btt">1.3 </td>
    <td class="tg-c3ow">0.744</td>
    <td class="tg-7btt">7.0 </td>
    <td class="tg-c3ow">0.742</td>
    <td class="tg-7btt">5.3 </td>
    <td class="tg-7btt">0.753</td>
    <td class="tg-7btt">2.1 </td>
    <td class="tg-c3ow">0.823</td>
    <td class="tg-7btt">3.7 </td>
    <td class="tg-c3ow">0.776</td>
  </tr>
</tbody></table>


**MOS Scores for Zero-Shot Cloned Voice**

| **Model**       | **Prosody** | **Timbre** | **Quality** |  **AVG**  |
|-----------------|:-----------:|:----------:|:-----------:|:---------:|
| **CosyVoice 2** |    3.67     |    4.05    |    3.73     |   3.81    |
| **F5TTS**       |    3.56     |    3.88    |    3.56     |   3.66    |
| **Fishspeech**  |    3.40     |    3.63    |    3.69     |   3.57    |
| **FireRedTTS**  |    3.79     |    3.72    |    3.60     |   3.70    |
| **XTTS**        |    3.23     |    2.99    |    3.10     |   3.11    |
| **IndexTTS**    |    **3.79**     |    **4.20**    |    **4.05**     |   **4.01**    |


## 📚 Citation

🌟 If you find our work helpful, please leave us a star and cite our paper.

```
@misc{deng2025indexttsindustriallevelcontrollableefficient,
      title={IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System}, 
      author={Wei Deng and Siyi Zhou and Jingchen Shu and Jinchao Wang and Lu Wang},
      year={2025},
      eprint={2502.05512},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2502.05512}, 
}
```
