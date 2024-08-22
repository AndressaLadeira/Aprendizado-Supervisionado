<title>Trabalho Regressão</title>
<style>
code{white-space: pre-wrap;}
span.smallcaps{font-variant: small-caps;}
div.columns{display: flex; gap: min(4vw, 1.5em);}
div.column{flex: auto; overflow-x: auto;}
div.hanging-indent{margin-left: 1.5em; text-indent: -1.5em;}
ul.task-list{list-style: none;}
ul.task-list li input[type="checkbox"] {
  width: 0.8em;
  margin: 0 0.8em 0.2em -1em; /* quarto-specific, see https://github.com/quarto-dev/quarto-cli/issues/4556 */ 
  vertical-align: middle;
}
/* CSS for syntax highlighting */
pre > code.sourceCode { white-space: pre; position: relative; }
pre > code.sourceCode > span { line-height: 1.25; }
pre > code.sourceCode > span:empty { height: 1.2em; }
.sourceCode { overflow: visible; }
code.sourceCode > span { color: inherit; text-decoration: inherit; }
div.sourceCode { margin: 1em 0; }
pre.sourceCode { margin: 0; }
@media screen {
div.sourceCode { overflow: auto; }
}
@media print {
pre > code.sourceCode { white-space: pre-wrap; }
pre > code.sourceCode > span { text-indent: -5em; padding-left: 5em; }
}
pre.numberSource code
  { counter-reset: source-line 0; }
pre.numberSource code > span
  { position: relative; left: -4em; counter-increment: source-line; }
pre.numberSource code > span > a:first-child::before
  { content: counter(source-line);
    position: relative; left: -1em; text-align: right; vertical-align: baseline;
    border: none; display: inline-block;
    -webkit-touch-callout: none; -webkit-user-select: none;
    -khtml-user-select: none; -moz-user-select: none;
    -ms-user-select: none; user-select: none;
    padding: 0 4px; width: 4em;
  }
pre.numberSource { margin-left: 3em;  padding-left: 4px; }
div.sourceCode
  {   }
@media screen {
pre > code.sourceCode > span > a:first-child::before { text-decoration: underline; }
}
</style>


<script src="Trabalho Aprendizado Supervisionado - Regressão_files/libs/clipboard/clipboard.min.js"></script>
<script src="Trabalho Aprendizado Supervisionado - Regressão_files/libs/quarto-html/quarto.js"></script>
<script src="Trabalho Aprendizado Supervisionado - Regressão_files/libs/quarto-html/popper.min.js"></script>
<script src="Trabalho Aprendizado Supervisionado - Regressão_files/libs/quarto-html/tippy.umd.min.js"></script>
<script src="Trabalho Aprendizado Supervisionado - Regressão_files/libs/quarto-html/anchor.min.js"></script>
<link href="Trabalho Aprendizado Supervisionado - Regressão_files/libs/quarto-html/tippy.css" rel="stylesheet">
<link href="Trabalho Aprendizado Supervisionado - Regressão_files/libs/quarto-html/quarto-syntax-highlighting.css" rel="stylesheet" id="quarto-text-highlighting-styles">
<script src="Trabalho Aprendizado Supervisionado - Regressão_files/libs/bootstrap/bootstrap.min.js"></script>
<link href="Trabalho Aprendizado Supervisionado - Regressão_files/libs/bootstrap/bootstrap-icons.css" rel="stylesheet">
<link href="Trabalho Aprendizado Supervisionado - Regressão_files/libs/bootstrap/bootstrap.min.css" rel="stylesheet" id="quarto-bootstrap" data-mode="light">


</head>

<body class="fullcontent">

<div id="quarto-content" class="page-columns page-rows-contents page-layout-article">

<main class="content" id="quarto-document-content">

<header id="title-block-header" class="quarto-title-block default">
<div class="quarto-title">
<h1 class="title">Trabalho Regressão</h1>
</div>



<div class="quarto-title-meta">

    <div>
    <div class="quarto-title-meta-heading">Author</div>
    <div class="quarto-title-meta-contents">
             <p>Aprendizado Supervisionado </p>
          </div>
  </div>
    
  
    
  </div>
  


</header>


<section id="trabalho-de-regressão-previsão-do-valor-de-apartmentos-em-minas-gerais-utilizando-o-pacote-tidymodels" class="level2">
<h2 class="anchored" data-anchor-id="trabalho-de-regressão-previsão-do-valor-de-apartmentos-em-minas-gerais-utilizando-o-pacote-tidymodels">Trabalho de Regressão: Previsão do valor de apartmentos em Minas Gerais utilizando o pacote ‘tidymodels’</h2>
<p>Carregando pacotes.</p>
<div class="cell">
<div class="sourceCode cell-code" id="cb1"><pre class="sourceCode r code-with-copy"><code class="sourceCode r"><span id="cb1-1"><a href="#cb1-1" aria-hidden="true" tabindex="-1"></a><span class="fu">library</span>(tidymodels)</span>
<span id="cb1-2"><a href="#cb1-2" aria-hidden="true" tabindex="-1"></a><span class="fu">library</span>(modelsummary)</span>
<span id="cb1-3"><a href="#cb1-3" aria-hidden="true" tabindex="-1"></a><span class="fu">library</span>(finetune)</span>
<span id="cb1-4"><a href="#cb1-4" aria-hidden="true" tabindex="-1"></a><span class="fu">library</span>(dplyr)</span>
<span id="cb1-5"><a href="#cb1-5" aria-hidden="true" tabindex="-1"></a><span class="fu">library</span>(baguette)</span>
<span id="cb1-6"><a href="#cb1-6" aria-hidden="true" tabindex="-1"></a><span class="fu">library</span>(readxl)</span>
<span id="cb1-7"><a href="#cb1-7" aria-hidden="true" tabindex="-1"></a><span class="fu">library</span>(glmnet)</span>
<span id="cb1-8"><a href="#cb1-8" aria-hidden="true" tabindex="-1"></a><span class="fu">library</span>(ranger)</span></code><button title="Copy to Clipboard" class="code-copy-button"><i class="bi"></i></button></pre></div>
</div>
<p>Leitura de dados.</p>
<div class="cell">
<div class="sourceCode cell-code" id="cb2"><pre class="sourceCode r code-with-copy"><code class="sourceCode r"><span id="cb2-1"><a href="#cb2-1" aria-hidden="true" tabindex="-1"></a>dados <span class="ot">&lt;-</span> <span class="fu">read_excel</span>(<span class="st">"C:/Users/andre/Downloads/Trab. Aprendizado Supervisionado.xlsx"</span>) <span class="sc">|&gt;</span></span>
<span id="cb2-2"><a href="#cb2-2" aria-hidden="true" tabindex="-1"></a>  <span class="fu">mutate</span>(<span class="fu">across</span>(<span class="fu">where</span>(is.character), as.factor),  <span class="co"># Converte variáveis categóricas para fatores</span></span>
<span id="cb2-3"><a href="#cb2-3" aria-hidden="true" tabindex="-1"></a>  <span class="at">Valor =</span> <span class="fu">as.numeric</span>(Valor))  </span>
<span id="cb2-4"><a href="#cb2-4" aria-hidden="true" tabindex="-1"></a>dados <span class="ot">&lt;-</span> <span class="fu">na.omit</span>(dados)</span></code><button title="Copy to Clipboard" class="code-copy-button"><i class="bi"></i></button></pre></div>
</div>
<p>Verificando a estrutura dos dados.</p>
<div class="cell">
<div class="sourceCode cell-code" id="cb3"><pre class="sourceCode r code-with-copy"><code class="sourceCode r"><span id="cb3-1"><a href="#cb3-1" aria-hidden="true" tabindex="-1"></a>dados <span class="ot">&lt;-</span> dados <span class="sc">|&gt;</span> </span>
<span id="cb3-2"><a href="#cb3-2" aria-hidden="true" tabindex="-1"></a>  <span class="fu">mutate</span>(<span class="fu">across</span>(<span class="fu">where</span>(is.character), as.factor))  <span class="co"># Convertendo variáveis categóricas em fatores</span></span>
<span id="cb3-3"><a href="#cb3-3" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb3-4"><a href="#cb3-4" aria-hidden="true" tabindex="-1"></a>dados <span class="sc">|&gt;</span> <span class="fu">glimpse</span>()</span></code><button title="Copy to Clipboard" class="code-copy-button"><i class="bi"></i></button></pre></div>
<div class="cell-output cell-output-stdout">
<pre><code>Rows: 632
Columns: 16
$ Apartamento &lt;dbl&gt; 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,…
$ Cidade      &lt;fct&gt; Barbacena, Barbacena, Barbacena, Barbacena, Barbacena, Bar…
$ Bairro      &lt;fct&gt; "Santa Tereza", "Boa Morte", "Serra Verde", "Centro", "São…
$ Area        &lt;fct&gt; 90.0, 74.0, 54.0, 88.0, 210.0, 133.0, 90.0, 19.0, 104.0, 2…
$ Valor       &lt;dbl&gt; 57, 52, 25, 63, 103, 94, 67, 5, 70, 97, 50, 99, 24, 46, 10…
$ Quartos     &lt;dbl&gt; 2, 2, 2, 3, 3, 2, 3, 1, 3, 3, 2, 3, 2, 2, 4, 3, 2, 3, 4, 3…
$ Banheiros   &lt;dbl&gt; 2, 2, 1, 3, 3, 2, 2, 1, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2…
$ Vaga        &lt;dbl&gt; 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 2, 1, 1, 2, 0, 1, 0, 0, 2…
$ Varanda     &lt;fct&gt; 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0…
$ Suite       &lt;dbl&gt; 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 2, 1, 0, 1, 1, 1…
$ AreaGourmet &lt;fct&gt; S, N, N, S, S, N, N, N, N, S, N, N, N, N, N, N, N, N, N, N…
$ Terraco     &lt;fct&gt; N, N, N, N, S, N, N, N, N, S, N, N, N, N, N, S, N, N, N, N…
$ Sala        &lt;dbl&gt; 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1…
$ Copa        &lt;fct&gt; S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, S, N, N…
$ Piscina     &lt;fct&gt; N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N…
$ Link        &lt;fct&gt; "https://www.zapimoveis.com.br/imovel/venda-apartamento-2-…</code></pre>
</div>
</div>
<p>Estatística descritiva.</p>
<div class="cell">
<div class="sourceCode cell-code" id="cb5"><pre class="sourceCode r code-with-copy"><code class="sourceCode r"><span id="cb5-1"><a href="#cb5-1" aria-hidden="true" tabindex="-1"></a><span class="fu">datasummary_skim</span>(dados)</span></code><button title="Copy to Clipboard" class="code-copy-button"><i class="bi"></i></button></pre></div>
<div class="cell-output cell-output-stderr">
<pre><code>Warning: These variables were omitted because they include more than 50 levels:
Bairro, Area, Link.</code></pre>
</div>
<div class="cell-output-display">
 

  
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>tinytable_l0jfhpeddufnncyzcok6</title>
    <style>
.table td.tinytable_css_romjfll2i4uiahcvkf3o, .table th.tinytable_css_romjfll2i4uiahcvkf3o {    border-bottom: solid 0.1em #d3d8dc; }
.table td.tinytable_css_f9osnyvu7gshniruikah, .table th.tinytable_css_f9osnyvu7gshniruikah {    border-top: solid 0.1em black; }
    </style>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async="" src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script>
    MathJax = {
      tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']]
      },
      svg: {
        fontCache: 'global'
      }
    };
    </script>
  

  
    <div class="container">
      <table class="table table-borderless" id="tinytable_l0jfhpeddufnncyzcok6" style="width: auto; margin-left: auto; margin-right: auto;" data-quarto-disable-processing="true">
        <thead>
        
              <tr>
                <th scope="col"> </th>
                <th scope="col">Unique</th>
                <th scope="col">Missing Pct.</th>
                <th scope="col">Mean</th>
                <th scope="col">SD</th>
                <th scope="col">Min</th>
                <th scope="col">Median</th>
                <th scope="col">Max</th>
                <th scope="col">Histogram</th>
              </tr>
        </thead>
        
        <tbody>
                <tr>
                  <td>Apartamento</td>
                  <td>632</td>
                  <td>0</td>
                  <td>316.5</td>
                  <td>182.6</td>
                  <td>1.0</td>
                  <td>316.5</td>
                  <td>632.0</td>
                  <td><img src="./tinytable_assets/id1119la7ww4rhtymutki0.png" style="height: 1em;"></td>
                </tr>
                <tr>
                  <td>Valor</td>
                  <td>278</td>
                  <td>0</td>
                  <td>127.5</td>
                  <td>76.2</td>
                  <td>1.0</td>
                  <td>114.5</td>
                  <td>278.0</td>
                  <td><img src="./tinytable_assets/idgztt5iohu4ro13us9fj5.png" style="height: 1em;"></td>
                </tr>
                <tr>
                  <td>Quartos</td>
                  <td>6</td>
                  <td>0</td>
                  <td>2.7</td>
                  <td>0.8</td>
                  <td>1.0</td>
                  <td>3.0</td>
                  <td>6.0</td>
                  <td><img src="./tinytable_assets/id92thhg8sive5gfyl8n74.png" style="height: 1em;"></td>
                </tr>
                <tr>
                  <td>Banheiros</td>
                  <td>6</td>
                  <td>0</td>
                  <td>1.8</td>
                  <td>0.9</td>
                  <td>1.0</td>
                  <td>2.0</td>
                  <td>6.0</td>
                  <td><img src="./tinytable_assets/iddc6q7ilb5am12sm4m1g5.png" style="height: 1em;"></td>
                </tr>
                <tr>
                  <td>Vaga</td>
                  <td>8</td>
                  <td>0</td>
                  <td>1.4</td>
                  <td>1.1</td>
                  <td>0.0</td>
                  <td>1.0</td>
                  <td>12.0</td>
                  <td><img src="./tinytable_assets/idmnnd5vbjetthpmj0nzzx.png" style="height: 1em;"></td>
                </tr>
                <tr>
                  <td>Suite</td>
                  <td>5</td>
                  <td>0</td>
                  <td>0.8</td>
                  <td>0.7</td>
                  <td>0.0</td>
                  <td>1.0</td>
                  <td>4.0</td>
                  <td><img src="./tinytable_assets/id2hw85e8ay0wxdpc28gzh.png" style="height: 1em;"></td>
                </tr>
                <tr>
                  <td>Sala</td>
                  <td>5</td>
                  <td>0</td>
                  <td>1.0</td>
                  <td>0.4</td>
                  <td>0.0</td>
                  <td>1.0</td>
                  <td>5.0</td>
                  <td><img src="./tinytable_assets/id9y1odjmucr0mjiizsnle.png" style="height: 1em;"></td>
                </tr>
                <tr>
                  <td> </td>
                  <td>  </td>
                  <td>N</td>
                  <td>%</td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                </tr>
                <tr>
                  <td>Cidade</td>
                  <td>Barbacena</td>
                  <td>100</td>
                  <td>15.8</td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                </tr>
                <tr>
                  <td></td>
                  <td>Congonhas</td>
                  <td>100</td>
                  <td>15.8</td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                </tr>
                <tr>
                  <td></td>
                  <td>Conselheiro Lafaiete</td>
                  <td>100</td>
                  <td>15.8</td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                </tr>
                <tr>
                  <td></td>
                  <td>Juiz de Fora</td>
                  <td>100</td>
                  <td>15.8</td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                </tr>
                <tr>
                  <td></td>
                  <td>Lavras</td>
                  <td>32</td>
                  <td>5.1</td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                </tr>
                <tr>
                  <td></td>
                  <td>São João del Rei</td>
                  <td>100</td>
                  <td>15.8</td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                </tr>
                <tr>
                  <td></td>
                  <td>Ubá</td>
                  <td>100</td>
                  <td>15.8</td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                </tr>
                <tr>
                  <td>Varanda</td>
                  <td>0.0</td>
                  <td>296</td>
                  <td>46.8</td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                </tr>
                <tr>
                  <td></td>
                  <td>1.0</td>
                  <td>210</td>
                  <td>33.2</td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                </tr>
                <tr>
                  <td></td>
                  <td>2.0</td>
                  <td>24</td>
                  <td>3.8</td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                </tr>
                <tr>
                  <td></td>
                  <td>3.0</td>
                  <td>1</td>
                  <td>0.2</td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                </tr>
                <tr>
                  <td></td>
                  <td>4.0</td>
                  <td>1</td>
                  <td>0.2</td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                </tr>
                <tr>
                  <td></td>
                  <td>n</td>
                  <td>1</td>
                  <td>0.2</td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                </tr>
                <tr>
                  <td></td>
                  <td>N</td>
                  <td>38</td>
                  <td>6.0</td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                </tr>
                <tr>
                  <td></td>
                  <td>S</td>
                  <td>61</td>
                  <td>9.7</td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                </tr>
                <tr>
                  <td>AreaGourmet</td>
                  <td>0.0</td>
                  <td>88</td>
                  <td>13.9</td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                </tr>
                <tr>
                  <td></td>
                  <td>1.0</td>
                  <td>12</td>
                  <td>1.9</td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                </tr>
                <tr>
                  <td></td>
                  <td>N</td>
                  <td>419</td>
                  <td>66.3</td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                </tr>
                <tr>
                  <td></td>
                  <td>S</td>
                  <td>113</td>
                  <td>17.9</td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                </tr>
                <tr>
                  <td>Terraco</td>
                  <td>0.0</td>
                  <td>98</td>
                  <td>15.5</td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                </tr>
                <tr>
                  <td></td>
                  <td>1.0</td>
                  <td>2</td>
                  <td>0.3</td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                </tr>
                <tr>
                  <td></td>
                  <td>N</td>
                  <td>461</td>
                  <td>72.9</td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                </tr>
                <tr>
                  <td></td>
                  <td>S</td>
                  <td>71</td>
                  <td>11.2</td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                </tr>
                <tr>
                  <td>Copa</td>
                  <td>0.0</td>
                  <td>94</td>
                  <td>14.9</td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                </tr>
                <tr>
                  <td></td>
                  <td>1.0</td>
                  <td>7</td>
                  <td>1.1</td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                </tr>
                <tr>
                  <td></td>
                  <td>N</td>
                  <td>403</td>
                  <td>63.8</td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                </tr>
                <tr>
                  <td></td>
                  <td>S</td>
                  <td>128</td>
                  <td>20.3</td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                </tr>
                <tr>
                  <td>Piscina</td>
                  <td>0.0</td>
                  <td>96</td>
                  <td>15.2</td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                </tr>
                <tr>
                  <td></td>
                  <td>1.0</td>
                  <td>4</td>
                  <td>0.6</td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                </tr>
                <tr>
                  <td></td>
                  <td>N</td>
                  <td>504</td>
                  <td>79.7</td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                </tr>
                <tr>
                  <td></td>
                  <td>S</td>
                  <td>28</td>
                  <td>4.4</td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                </tr>
        </tbody>
      </table>
    </div>

    <script>
      function styleCell_tinytable_7ofwuhnotay4gsm9tbbt(i, j, css_id) {
        var table = document.getElementById("tinytable_l0jfhpeddufnncyzcok6");
        table.rows[i].cells[j].classList.add(css_id);
      }
      function insertSpanRow(i, colspan, content) {
        var table = document.getElementById('tinytable_l0jfhpeddufnncyzcok6');
        var newRow = table.insertRow(i);
        var newCell = newRow.insertCell(0);
        newCell.setAttribute("colspan", colspan);
        // newCell.innerText = content;
        // this may be unsafe, but innerText does not interpret <br>
        newCell.innerHTML = content;
      }
      function spanCell_tinytable_7ofwuhnotay4gsm9tbbt(i, j, rowspan, colspan) {
        var table = document.getElementById("tinytable_l0jfhpeddufnncyzcok6");
        const targetRow = table.rows[i];
        const targetCell = targetRow.cells[j];
        for (let r = 0; r < rowspan; r++) {
          // Only start deleting cells to the right for the first row (r == 0)
          if (r === 0) {
            // Delete cells to the right of the target cell in the first row
            for (let c = colspan - 1; c > 0; c--) {
              if (table.rows[i + r].cells[j + c]) {
                table.rows[i + r].deleteCell(j + c);
              }
            }
          }
          // For rows below the first, delete starting from the target column
          if (r > 0) {
            for (let c = colspan - 1; c >= 0; c--) {
              if (table.rows[i + r] && table.rows[i + r].cells[j]) {
                table.rows[i + r].deleteCell(j);
              }
            }
          }
        }
        // Set rowspan and colspan of the target cell
        targetCell.rowSpan = rowspan;
        targetCell.colSpan = colspan;
      }

window.addEventListener('load', function () { styleCell_tinytable_7ofwuhnotay4gsm9tbbt(0, 0, 'tinytable_css_romjfll2i4uiahcvkf3o') })
window.addEventListener('load', function () { styleCell_tinytable_7ofwuhnotay4gsm9tbbt(0, 1, 'tinytable_css_romjfll2i4uiahcvkf3o') })
window.addEventListener('load', function () { styleCell_tinytable_7ofwuhnotay4gsm9tbbt(0, 2, 'tinytable_css_romjfll2i4uiahcvkf3o') })
window.addEventListener('load', function () { styleCell_tinytable_7ofwuhnotay4gsm9tbbt(0, 3, 'tinytable_css_romjfll2i4uiahcvkf3o') })
window.addEventListener('load', function () { styleCell_tinytable_7ofwuhnotay4gsm9tbbt(0, 4, 'tinytable_css_romjfll2i4uiahcvkf3o') })
window.addEventListener('load', function () { styleCell_tinytable_7ofwuhnotay4gsm9tbbt(0, 5, 'tinytable_css_romjfll2i4uiahcvkf3o') })
window.addEventListener('load', function () { styleCell_tinytable_7ofwuhnotay4gsm9tbbt(0, 6, 'tinytable_css_romjfll2i4uiahcvkf3o') })
window.addEventListener('load', function () { styleCell_tinytable_7ofwuhnotay4gsm9tbbt(0, 7, 'tinytable_css_romjfll2i4uiahcvkf3o') })
window.addEventListener('load', function () { styleCell_tinytable_7ofwuhnotay4gsm9tbbt(0, 8, 'tinytable_css_romjfll2i4uiahcvkf3o') })
window.addEventListener('load', function () { styleCell_tinytable_7ofwuhnotay4gsm9tbbt(8, 0, 'tinytable_css_f9osnyvu7gshniruikah') })
window.addEventListener('load', function () { styleCell_tinytable_7ofwuhnotay4gsm9tbbt(8, 1, 'tinytable_css_f9osnyvu7gshniruikah') })
window.addEventListener('load', function () { styleCell_tinytable_7ofwuhnotay4gsm9tbbt(8, 2, 'tinytable_css_f9osnyvu7gshniruikah') })
window.addEventListener('load', function () { styleCell_tinytable_7ofwuhnotay4gsm9tbbt(8, 3, 'tinytable_css_f9osnyvu7gshniruikah') })
window.addEventListener('load', function () { styleCell_tinytable_7ofwuhnotay4gsm9tbbt(8, 4, 'tinytable_css_f9osnyvu7gshniruikah') })
window.addEventListener('load', function () { styleCell_tinytable_7ofwuhnotay4gsm9tbbt(8, 5, 'tinytable_css_f9osnyvu7gshniruikah') })
window.addEventListener('load', function () { styleCell_tinytable_7ofwuhnotay4gsm9tbbt(8, 6, 'tinytable_css_f9osnyvu7gshniruikah') })
window.addEventListener('load', function () { styleCell_tinytable_7ofwuhnotay4gsm9tbbt(8, 7, 'tinytable_css_f9osnyvu7gshniruikah') })
window.addEventListener('load', function () { styleCell_tinytable_7ofwuhnotay4gsm9tbbt(8, 8, 'tinytable_css_f9osnyvu7gshniruikah') })
    </script>

  


</div>
</div>
<p>Separando dados de treino e teste, com 75% dos dados para treino.</p>
<div class="cell">
<div class="sourceCode cell-code" id="cb7"><pre class="sourceCode r code-with-copy"><code class="sourceCode r"><span id="cb7-1"><a href="#cb7-1" aria-hidden="true" tabindex="-1"></a><span class="fu">set.seed</span>(<span class="dv">16</span>)</span>
<span id="cb7-2"><a href="#cb7-2" aria-hidden="true" tabindex="-1"></a>dados_split <span class="ot">&lt;-</span> <span class="fu">initial_split</span>(dados, </span>
<span id="cb7-3"><a href="#cb7-3" aria-hidden="true" tabindex="-1"></a>                             <span class="at">prop =</span> <span class="fl">0.75</span>,</span>
<span id="cb7-4"><a href="#cb7-4" aria-hidden="true" tabindex="-1"></a>                             <span class="at">strata =</span> Cidade)  <span class="co"># Supondo que a cidade seja relevante para estratificação</span></span>
<span id="cb7-5"><a href="#cb7-5" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb7-6"><a href="#cb7-6" aria-hidden="true" tabindex="-1"></a>dados_train <span class="ot">&lt;-</span> <span class="fu">training</span>(dados_split)</span>
<span id="cb7-7"><a href="#cb7-7" aria-hidden="true" tabindex="-1"></a>dados_test  <span class="ot">&lt;-</span> <span class="fu">testing</span>(dados_split)</span></code><button title="Copy to Clipboard" class="code-copy-button"><i class="bi"></i></button></pre></div>
</div>
<p>Definindo a receita com normalização das variáveis numéricas e dummificação das variáveis categóricas.</p>
<div class="cell">
<div class="sourceCode cell-code" id="cb8"><pre class="sourceCode r code-with-copy"><code class="sourceCode r"><span id="cb8-1"><a href="#cb8-1" aria-hidden="true" tabindex="-1"></a>normalized_rec <span class="ot">&lt;-</span> </span>
<span id="cb8-2"><a href="#cb8-2" aria-hidden="true" tabindex="-1"></a>  <span class="fu">recipe</span>(Valor <span class="sc">~</span> ., <span class="at">data =</span> dados_train) <span class="sc">|&gt;</span> </span>
<span id="cb8-3"><a href="#cb8-3" aria-hidden="true" tabindex="-1"></a>  <span class="fu">step_normalize</span>(<span class="fu">all_numeric_predictors</span>()) <span class="sc">|&gt;</span></span>
<span id="cb8-4"><a href="#cb8-4" aria-hidden="true" tabindex="-1"></a>  <span class="fu">step_dummy</span>(<span class="fu">all_nominal_predictors</span>())</span></code><button title="Copy to Clipboard" class="code-copy-button"><i class="bi"></i></button></pre></div>
</div>
<p>Definindo os modelos de regressão.</p>
<div class="cell">
<div class="sourceCode cell-code" id="cb9"><pre class="sourceCode r code-with-copy"><code class="sourceCode r"><span id="cb9-1"><a href="#cb9-1" aria-hidden="true" tabindex="-1"></a>linear_reg_spec <span class="ot">&lt;-</span> </span>
<span id="cb9-2"><a href="#cb9-2" aria-hidden="true" tabindex="-1"></a>  <span class="fu">linear_reg</span>(<span class="at">penalty =</span> <span class="fu">tune</span>(), <span class="at">mixture =</span> <span class="fu">tune</span>()) <span class="sc">|&gt;</span> </span>
<span id="cb9-3"><a href="#cb9-3" aria-hidden="true" tabindex="-1"></a>  <span class="fu">set_engine</span>(<span class="st">"glmnet"</span>)</span>
<span id="cb9-4"><a href="#cb9-4" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb9-5"><a href="#cb9-5" aria-hidden="true" tabindex="-1"></a>tree_spec <span class="ot">&lt;-</span> <span class="fu">decision_tree</span>(<span class="at">tree_depth =</span> <span class="fu">tune</span>(), <span class="at">min_n =</span> <span class="fu">tune</span>(), <span class="at">cost_complexity =</span> <span class="fu">tune</span>()) <span class="sc">|&gt;</span> </span>
<span id="cb9-6"><a href="#cb9-6" aria-hidden="true" tabindex="-1"></a>  <span class="fu">set_engine</span>(<span class="st">"rpart"</span>) <span class="sc">|&gt;</span> </span>
<span id="cb9-7"><a href="#cb9-7" aria-hidden="true" tabindex="-1"></a>  <span class="fu">set_mode</span>(<span class="st">"regression"</span>)</span>
<span id="cb9-8"><a href="#cb9-8" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb9-9"><a href="#cb9-9" aria-hidden="true" tabindex="-1"></a>rforest_spec <span class="ot">&lt;-</span> <span class="fu">rand_forest</span>(<span class="at">mtry =</span> <span class="fu">tune</span>(), <span class="at">min_n =</span> <span class="fu">tune</span>(), <span class="at">trees =</span> <span class="fu">tune</span>()) <span class="sc">|&gt;</span> </span>
<span id="cb9-10"><a href="#cb9-10" aria-hidden="true" tabindex="-1"></a>  <span class="fu">set_engine</span>(<span class="st">"ranger"</span>) <span class="sc">|&gt;</span> </span>
<span id="cb9-11"><a href="#cb9-11" aria-hidden="true" tabindex="-1"></a>  <span class="fu">set_mode</span>(<span class="st">"regression"</span>)</span></code><button title="Copy to Clipboard" class="code-copy-button"><i class="bi"></i></button></pre></div>
</div>
<p>Definindo o workflow.</p>
<div class="cell">
<div class="sourceCode cell-code" id="cb10"><pre class="sourceCode r code-with-copy"><code class="sourceCode r"><span id="cb10-1"><a href="#cb10-1" aria-hidden="true" tabindex="-1"></a>normalized <span class="ot">&lt;-</span> </span>
<span id="cb10-2"><a href="#cb10-2" aria-hidden="true" tabindex="-1"></a>  <span class="fu">workflow_set</span>(</span>
<span id="cb10-3"><a href="#cb10-3" aria-hidden="true" tabindex="-1"></a>    <span class="at">preproc =</span> <span class="fu">list</span>(<span class="at">normalized =</span> normalized_rec), </span>
<span id="cb10-4"><a href="#cb10-4" aria-hidden="true" tabindex="-1"></a>    <span class="at">models =</span> <span class="fu">list</span>(<span class="at">linear_reg =</span> linear_reg_spec,</span>
<span id="cb10-5"><a href="#cb10-5" aria-hidden="true" tabindex="-1"></a>                  <span class="at">tree =</span> tree_spec,</span>
<span id="cb10-6"><a href="#cb10-6" aria-hidden="true" tabindex="-1"></a>                  <span class="at">rforest =</span> rforest_spec)</span>
<span id="cb10-7"><a href="#cb10-7" aria-hidden="true" tabindex="-1"></a>  )</span></code><button title="Copy to Clipboard" class="code-copy-button"><i class="bi"></i></button></pre></div>
</div>
<p>Simplificando os nomes dos modelos.</p>
<div class="cell">
<div class="sourceCode cell-code" id="cb11"><pre class="sourceCode r code-with-copy"><code class="sourceCode r"><span id="cb11-1"><a href="#cb11-1" aria-hidden="true" tabindex="-1"></a>all_workflows <span class="ot">&lt;-</span> </span>
<span id="cb11-2"><a href="#cb11-2" aria-hidden="true" tabindex="-1"></a>  <span class="fu">bind_rows</span>(normalized) <span class="sc">|&gt;</span> </span>
<span id="cb11-3"><a href="#cb11-3" aria-hidden="true" tabindex="-1"></a>  <span class="fu">mutate</span>(<span class="at">wflow_id =</span> <span class="fu">gsub</span>(<span class="st">"(normalized_)"</span>, <span class="st">""</span>, wflow_id))</span></code><button title="Copy to Clipboard" class="code-copy-button"><i class="bi"></i></button></pre></div>
</div>
<p>Realizando grid search e validação cruzada.</p>
<div class="cell">
<div class="sourceCode cell-code" id="cb12"><pre class="sourceCode r code-with-copy"><code class="sourceCode r"><span id="cb12-1"><a href="#cb12-1" aria-hidden="true" tabindex="-1"></a>race_ctrl <span class="ot">&lt;-</span> <span class="fu">control_race</span>(</span>
<span id="cb12-2"><a href="#cb12-2" aria-hidden="true" tabindex="-1"></a>  <span class="at">save_pred =</span> <span class="cn">TRUE</span>,</span>
<span id="cb12-3"><a href="#cb12-3" aria-hidden="true" tabindex="-1"></a>  <span class="at">parallel_over =</span> <span class="st">"everything"</span>,</span>
<span id="cb12-4"><a href="#cb12-4" aria-hidden="true" tabindex="-1"></a>  <span class="at">save_workflow =</span> <span class="cn">TRUE</span></span>
<span id="cb12-5"><a href="#cb12-5" aria-hidden="true" tabindex="-1"></a>)</span>
<span id="cb12-6"><a href="#cb12-6" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb12-7"><a href="#cb12-7" aria-hidden="true" tabindex="-1"></a><span class="fu">set.seed</span>(<span class="dv">1503</span>)</span>
<span id="cb12-8"><a href="#cb12-8" aria-hidden="true" tabindex="-1"></a>race_results <span class="ot">&lt;-</span> all_workflows <span class="sc">|&gt;</span></span>
<span id="cb12-9"><a href="#cb12-9" aria-hidden="true" tabindex="-1"></a>  <span class="fu">workflow_map</span>(</span>
<span id="cb12-10"><a href="#cb12-10" aria-hidden="true" tabindex="-1"></a>    <span class="st">"tune_race_anova"</span>,</span>
<span id="cb12-11"><a href="#cb12-11" aria-hidden="true" tabindex="-1"></a>    <span class="at">seed =</span> <span class="dv">1503</span>,</span>
<span id="cb12-12"><a href="#cb12-12" aria-hidden="true" tabindex="-1"></a>    <span class="at">resamples =</span> <span class="fu">vfold_cv</span>(<span class="at">v =</span> <span class="dv">10</span>, dados_train, <span class="at">repeats =</span> <span class="dv">2</span>),</span>
<span id="cb12-13"><a href="#cb12-13" aria-hidden="true" tabindex="-1"></a>    <span class="at">grid =</span> <span class="dv">25</span>,</span>
<span id="cb12-14"><a href="#cb12-14" aria-hidden="true" tabindex="-1"></a>    <span class="at">control =</span> race_ctrl</span>
<span id="cb12-15"><a href="#cb12-15" aria-hidden="true" tabindex="-1"></a>  )</span></code><button title="Copy to Clipboard" class="code-copy-button"><i class="bi"></i></button></pre></div>
<div class="cell-output cell-output-stderr">
<pre><code>Warning: package 'rlang' was built under R version 4.3.3</code></pre>
</div>
</div>
<p>Extraindo métricas de desempenho.</p>
<div class="cell">
<div class="sourceCode cell-code" id="cb14"><pre class="sourceCode r code-with-copy"><code class="sourceCode r"><span id="cb14-1"><a href="#cb14-1" aria-hidden="true" tabindex="-1"></a><span class="fu">collect_metrics</span>(race_results) <span class="sc">|&gt;</span> </span>
<span id="cb14-2"><a href="#cb14-2" aria-hidden="true" tabindex="-1"></a>  <span class="fu">filter</span>(.metric <span class="sc">==</span> <span class="st">"rmse"</span>) <span class="sc">|&gt;</span></span>
<span id="cb14-3"><a href="#cb14-3" aria-hidden="true" tabindex="-1"></a>  <span class="fu">arrange</span>(mean)</span></code><button title="Copy to Clipboard" class="code-copy-button"><i class="bi"></i></button></pre></div>
<div class="cell-output cell-output-stdout">
<pre><code># A tibble: 25 × 9
   wflow_id .config         preproc model .metric .estimator  mean     n std_err
   &lt;chr&gt;    &lt;chr&gt;           &lt;chr&gt;   &lt;chr&gt; &lt;chr&gt;   &lt;chr&gt;      &lt;dbl&gt; &lt;int&gt;   &lt;dbl&gt;
 1 rforest  Preprocessor1_… recipe  rand… rmse    standard    28.5    20   0.845
 2 rforest  Preprocessor1_… recipe  rand… rmse    standard    28.6    20   0.819
 3 rforest  Preprocessor1_… recipe  rand… rmse    standard    28.6    20   0.848
 4 rforest  Preprocessor1_… recipe  rand… rmse    standard    28.6    20   0.880
 5 rforest  Preprocessor1_… recipe  rand… rmse    standard    28.6    20   0.804
 6 rforest  Preprocessor1_… recipe  rand… rmse    standard    28.6    20   0.763
 7 tree     Preprocessor1_… recipe  deci… rmse    standard    31.0    20   0.987
 8 tree     Preprocessor1_… recipe  deci… rmse    standard    31.3    20   0.932
 9 tree     Preprocessor1_… recipe  deci… rmse    standard    31.4    20   1.05 
10 tree     Preprocessor1_… recipe  deci… rmse    standard    31.5    20   0.998
# ℹ 15 more rows</code></pre>
</div>
</div>
<p>Visualizando o desempenho dos modelos.</p>
<div class="cell">
<div class="sourceCode cell-code" id="cb16"><pre class="sourceCode r code-with-copy"><code class="sourceCode r"><span id="cb16-1"><a href="#cb16-1" aria-hidden="true" tabindex="-1"></a>IC_rmse <span class="ot">&lt;-</span> <span class="fu">collect_metrics</span>(race_results) <span class="sc">|&gt;</span> </span>
<span id="cb16-2"><a href="#cb16-2" aria-hidden="true" tabindex="-1"></a>  <span class="fu">filter</span>(.metric <span class="sc">==</span> <span class="st">"rmse"</span>) <span class="sc">|&gt;</span> </span>
<span id="cb16-3"><a href="#cb16-3" aria-hidden="true" tabindex="-1"></a>  <span class="fu">group_by</span>(wflow_id) <span class="sc">|&gt;</span></span>
<span id="cb16-4"><a href="#cb16-4" aria-hidden="true" tabindex="-1"></a>  <span class="fu">arrange</span>(mean) <span class="sc">|&gt;</span> </span>
<span id="cb16-5"><a href="#cb16-5" aria-hidden="true" tabindex="-1"></a>  <span class="fu">ungroup</span>()</span>
<span id="cb16-6"><a href="#cb16-6" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb16-7"><a href="#cb16-7" aria-hidden="true" tabindex="-1"></a><span class="fu">ggplot</span>(IC_rmse, <span class="fu">aes</span>(<span class="at">x =</span> <span class="fu">factor</span>(wflow_id, <span class="at">levels =</span> <span class="fu">unique</span>(wflow_id)), <span class="at">y =</span> mean)) <span class="sc">+</span></span>
<span id="cb16-8"><a href="#cb16-8" aria-hidden="true" tabindex="-1"></a>  <span class="fu">geom_point</span>(<span class="at">stat=</span><span class="st">"identity"</span>, <span class="fu">aes</span>(<span class="at">color =</span> wflow_id), <span class="at">pch =</span> <span class="dv">1</span>) <span class="sc">+</span></span>
<span id="cb16-9"><a href="#cb16-9" aria-hidden="true" tabindex="-1"></a>  <span class="fu">geom_errorbar</span>(<span class="at">stat=</span><span class="st">"identity"</span>, <span class="fu">aes</span>(<span class="at">color =</span> wflow_id, </span>
<span id="cb16-10"><a href="#cb16-10" aria-hidden="true" tabindex="-1"></a>                                     <span class="at">ymin=</span>mean<span class="fl">-1.96</span><span class="sc">*</span>std_err,</span>
<span id="cb16-11"><a href="#cb16-11" aria-hidden="true" tabindex="-1"></a>                                     <span class="at">ymax=</span>mean<span class="fl">+1.96</span><span class="sc">*</span>std_err), <span class="at">width=</span>.<span class="dv">2</span>) <span class="sc">+</span> </span>
<span id="cb16-12"><a href="#cb16-12" aria-hidden="true" tabindex="-1"></a>  <span class="fu">labs</span>(<span class="at">y =</span> <span class="st">""</span>, <span class="at">x =</span> <span class="st">"method"</span>) <span class="sc">+</span> <span class="fu">theme_bw</span>() <span class="sc">+</span></span>
<span id="cb16-13"><a href="#cb16-13" aria-hidden="true" tabindex="-1"></a>  <span class="fu">theme</span>(<span class="at">legend.position =</span> <span class="st">"none"</span>,</span>
<span id="cb16-14"><a href="#cb16-14" aria-hidden="true" tabindex="-1"></a>        <span class="at">axis.text.x =</span> <span class="fu">element_text</span>(<span class="at">angle =</span> <span class="dv">90</span>, <span class="at">vjust =</span> <span class="fl">0.5</span>, <span class="at">hjust=</span><span class="dv">1</span>))</span></code><button title="Copy to Clipboard" class="code-copy-button"><i class="bi"></i></button></pre></div>
<div class="cell-output-display">
<div>
<figure class="figure">
<p><img src="Trabalho-Aprendizado-Supervisionado---Regressão_files/figure-html/unnamed-chunk-12-1.png" class="img-fluid figure-img" width="672"></p>
</figure>
</div>
</div>
</div>
</section>

</main>
<!-- /main column -->
<script id="quarto-html-after-body" type="application/javascript">
window.document.addEventListener("DOMContentLoaded", function (event) {
  const toggleBodyColorMode = (bsSheetEl) => {
    const mode = bsSheetEl.getAttribute("data-mode");
    const bodyEl = window.document.querySelector("body");
    if (mode === "dark") {
      bodyEl.classList.add("quarto-dark");
      bodyEl.classList.remove("quarto-light");
    } else {
      bodyEl.classList.add("quarto-light");
      bodyEl.classList.remove("quarto-dark");
    }
  }
  const toggleBodyColorPrimary = () => {
    const bsSheetEl = window.document.querySelector("link#quarto-bootstrap");
    if (bsSheetEl) {
      toggleBodyColorMode(bsSheetEl);
    }
  }
  toggleBodyColorPrimary();  
  const icon = "";
  const anchorJS = new window.AnchorJS();
  anchorJS.options = {
    placement: 'right',
    icon: icon
  };
  anchorJS.add('.anchored');
  const isCodeAnnotation = (el) => {
    for (const clz of el.classList) {
      if (clz.startsWith('code-annotation-')) {                     
        return true;
      }
    }
    return false;
  }
  const clipboard = new window.ClipboardJS('.code-copy-button', {
    text: function(trigger) {
      const codeEl = trigger.previousElementSibling.cloneNode(true);
      for (const childEl of codeEl.children) {
        if (isCodeAnnotation(childEl)) {
          childEl.remove();
        }
      }
      return codeEl.innerText;
    }
  });
  clipboard.on('success', function(e) {
    // button target
    const button = e.trigger;
    // don't keep focus
    button.blur();
    // flash "checked"
    button.classList.add('code-copy-button-checked');
    var currentTitle = button.getAttribute("title");
    button.setAttribute("title", "Copied!");
    let tooltip;
    if (window.bootstrap) {
      button.setAttribute("data-bs-toggle", "tooltip");
      button.setAttribute("data-bs-placement", "left");
      button.setAttribute("data-bs-title", "Copied!");
      tooltip = new bootstrap.Tooltip(button, 
        { trigger: "manual", 
          customClass: "code-copy-button-tooltip",
          offset: [0, -8]});
      tooltip.show();    
    }
    setTimeout(function() {
      if (tooltip) {
        tooltip.hide();
        button.removeAttribute("data-bs-title");
        button.removeAttribute("data-bs-toggle");
        button.removeAttribute("data-bs-placement");
      }
      button.setAttribute("title", currentTitle);
      button.classList.remove('code-copy-button-checked');
    }, 1000);
    // clear code selection
    e.clearSelection();
  });
    var localhostRegex = new RegExp(/^(?:http|https):\/\/localhost\:?[0-9]*\//);
    var mailtoRegex = new RegExp(/^mailto:/);
      var filterRegex = new RegExp('/' + window.location.host + '/');
    var isInternal = (href) => {
        return filterRegex.test(href) || localhostRegex.test(href) || mailtoRegex.test(href);
    }
    // Inspect non-navigation links and adorn them if external
 	var links = window.document.querySelectorAll('a[href]:not(.nav-link):not(.navbar-brand):not(.toc-action):not(.sidebar-link):not(.sidebar-item-toggle):not(.pagination-link):not(.no-external):not([aria-hidden]):not(.dropdown-item):not(.quarto-navigation-tool)');
    for (var i=0; i<links.length; i++) {
      const link = links[i];
      if (!isInternal(link.href)) {
        // undo the damage that might have been done by quarto-nav.js in the case of
        // links that we want to consider external
        if (link.dataset.originalHref !== undefined) {
          link.href = link.dataset.originalHref;
        }
      }
    }
  function tippyHover(el, contentFn, onTriggerFn, onUntriggerFn) {
    const config = {
      allowHTML: true,
      maxWidth: 500,
      delay: 100,
      arrow: false,
      appendTo: function(el) {
          return el.parentElement;
      },
      interactive: true,
      interactiveBorder: 10,
      theme: 'quarto',
      placement: 'bottom-start',
    };
    if (contentFn) {
      config.content = contentFn;
    }
    if (onTriggerFn) {
      config.onTrigger = onTriggerFn;
    }
    if (onUntriggerFn) {
      config.onUntrigger = onUntriggerFn;
    }
    window.tippy(el, config); 
  }
  const noterefs = window.document.querySelectorAll('a[role="doc-noteref"]');
  for (var i=0; i<noterefs.length; i++) {
    const ref = noterefs[i];
    tippyHover(ref, function() {
      // use id or data attribute instead here
      let href = ref.getAttribute('data-footnote-href') || ref.getAttribute('href');
      try { href = new URL(href).hash; } catch {}
      const id = href.replace(/^#\/?/, "");
      const note = window.document.getElementById(id);
      if (note) {
        return note.innerHTML;
      } else {
        return "";
      }
    });
  }
  const xrefs = window.document.querySelectorAll('a.quarto-xref');
  const processXRef = (id, note) => {
    // Strip column container classes
    const stripColumnClz = (el) => {
      el.classList.remove("page-full", "page-columns");
      if (el.children) {
        for (const child of el.children) {
          stripColumnClz(child);
        }
      }
    }
    stripColumnClz(note)
    if (id === null || id.startsWith('sec-')) {
      // Special case sections, only their first couple elements
      const container = document.createElement("div");
      if (note.children && note.children.length > 2) {
        container.appendChild(note.children[0].cloneNode(true));
        for (let i = 1; i < note.children.length; i++) {
          const child = note.children[i];
          if (child.tagName === "P" && child.innerText === "") {
            continue;
          } else {
            container.appendChild(child.cloneNode(true));
            break;
          }
        }
        if (window.Quarto?.typesetMath) {
          window.Quarto.typesetMath(container);
        }
        return container.innerHTML
      } else {
        if (window.Quarto?.typesetMath) {
          window.Quarto.typesetMath(note);
        }
        return note.innerHTML;
      }
    } else {
      // Remove any anchor links if they are present
      const anchorLink = note.querySelector('a.anchorjs-link');
      if (anchorLink) {
        anchorLink.remove();
      }
      if (window.Quarto?.typesetMath) {
        window.Quarto.typesetMath(note);
      }
      // TODO in 1.5, we should make sure this works without a callout special case
      if (note.classList.contains("callout")) {
        return note.outerHTML;
      } else {
        return note.innerHTML;
      }
    }
  }
  for (var i=0; i<xrefs.length; i++) {
    const xref = xrefs[i];
    tippyHover(xref, undefined, function(instance) {
      instance.disable();
      let url = xref.getAttribute('href');
      let hash = undefined; 
      if (url.startsWith('#')) {
        hash = url;
      } else {
        try { hash = new URL(url).hash; } catch {}
      }
      if (hash) {
        const id = hash.replace(/^#\/?/, "");
        const note = window.document.getElementById(id);
        if (note !== null) {
          try {
            const html = processXRef(id, note.cloneNode(true));
            instance.setContent(html);
          } finally {
            instance.enable();
            instance.show();
          }
        } else {
          // See if we can fetch this
          fetch(url.split('#')[0])
          .then(res => res.text())
          .then(html => {
            const parser = new DOMParser();
            const htmlDoc = parser.parseFromString(html, "text/html");
            const note = htmlDoc.getElementById(id);
            if (note !== null) {
              const html = processXRef(id, note);
              instance.setContent(html);
            } 
          }).finally(() => {
            instance.enable();
            instance.show();
          });
        }
      } else {
        // See if we can fetch a full url (with no hash to target)
        // This is a special case and we should probably do some content thinning / targeting
        fetch(url)
        .then(res => res.text())
        .then(html => {
          const parser = new DOMParser();
          const htmlDoc = parser.parseFromString(html, "text/html");
          const note = htmlDoc.querySelector('main.content');
          if (note !== null) {
            // This should only happen for chapter cross references
            // (since there is no id in the URL)
            // remove the first header
            if (note.children.length > 0 && note.children[0].tagName === "HEADER") {
              note.children[0].remove();
            }
            const html = processXRef(null, note);
            instance.setContent(html);
          } 
        }).finally(() => {
          instance.enable();
          instance.show();
        });
      }
    }, function(instance) {
    });
  }
      let selectedAnnoteEl;
      const selectorForAnnotation = ( cell, annotation) => {
        let cellAttr = 'data-code-cell="' + cell + '"';
        let lineAttr = 'data-code-annotation="' +  annotation + '"';
        const selector = 'span[' + cellAttr + '][' + lineAttr + ']';
        return selector;
      }
      const selectCodeLines = (annoteEl) => {
        const doc = window.document;
        const targetCell = annoteEl.getAttribute("data-target-cell");
        const targetAnnotation = annoteEl.getAttribute("data-target-annotation");
        const annoteSpan = window.document.querySelector(selectorForAnnotation(targetCell, targetAnnotation));
        const lines = annoteSpan.getAttribute("data-code-lines").split(",");
        const lineIds = lines.map((line) => {
          return targetCell + "-" + line;
        })
        let top = null;
        let height = null;
        let parent = null;
        if (lineIds.length > 0) {
            //compute the position of the single el (top and bottom and make a div)
            const el = window.document.getElementById(lineIds[0]);
            top = el.offsetTop;
            height = el.offsetHeight;
            parent = el.parentElement.parentElement;
          if (lineIds.length > 1) {
            const lastEl = window.document.getElementById(lineIds[lineIds.length - 1]);
            const bottom = lastEl.offsetTop + lastEl.offsetHeight;
            height = bottom - top;
          }
          if (top !== null && height !== null && parent !== null) {
            // cook up a div (if necessary) and position it 
            let div = window.document.getElementById("code-annotation-line-highlight");
            if (div === null) {
              div = window.document.createElement("div");
              div.setAttribute("id", "code-annotation-line-highlight");
              div.style.position = 'absolute';
              parent.appendChild(div);
            }
            div.style.top = top - 2 + "px";
            div.style.height = height + 4 + "px";
            div.style.left = 0;
            let gutterDiv = window.document.getElementById("code-annotation-line-highlight-gutter");
            if (gutterDiv === null) {
              gutterDiv = window.document.createElement("div");
              gutterDiv.setAttribute("id", "code-annotation-line-highlight-gutter");
              gutterDiv.style.position = 'absolute';
              const codeCell = window.document.getElementById(targetCell);
              const gutter = codeCell.querySelector('.code-annotation-gutter');
              gutter.appendChild(gutterDiv);
            }
            gutterDiv.style.top = top - 2 + "px";
            gutterDiv.style.height = height + 4 + "px";
          }
          selectedAnnoteEl = annoteEl;
        }
      };
      const unselectCodeLines = () => {
        const elementsIds = ["code-annotation-line-highlight", "code-annotation-line-highlight-gutter"];
        elementsIds.forEach((elId) => {
          const div = window.document.getElementById(elId);
          if (div) {
            div.remove();
          }
        });
        selectedAnnoteEl = undefined;
      };
        // Handle positioning of the toggle
    window.addEventListener(
      "resize",
      throttle(() => {
        elRect = undefined;
        if (selectedAnnoteEl) {
          selectCodeLines(selectedAnnoteEl);
        }
      }, 10)
    );
    function throttle(fn, ms) {
    let throttle = false;
    let timer;
      return (...args) => {
        if(!throttle) { // first call gets through
            fn.apply(this, args);
            throttle = true;
        } else { // all the others get throttled
            if(timer) clearTimeout(timer); // cancel #2
            timer = setTimeout(() => {
              fn.apply(this, args);
              timer = throttle = false;
            }, ms);
        }
      };
    }
      // Attach click handler to the DT
      const annoteDls = window.document.querySelectorAll('dt[data-target-cell]');
      for (const annoteDlNode of annoteDls) {
        annoteDlNode.addEventListener('click', (event) => {
          const clickedEl = event.target;
          if (clickedEl !== selectedAnnoteEl) {
            unselectCodeLines();
            const activeEl = window.document.querySelector('dt[data-target-cell].code-annotation-active');
            if (activeEl) {
              activeEl.classList.remove('code-annotation-active');
            }
            selectCodeLines(clickedEl);
            clickedEl.classList.add('code-annotation-active');
          } else {
            // Unselect the line
            unselectCodeLines();
            clickedEl.classList.remove('code-annotation-active');
          }
        });
      }
  const findCites = (el) => {
    const parentEl = el.parentElement;
    if (parentEl) {
      const cites = parentEl.dataset.cites;
      if (cites) {
        return {
          el,
          cites: cites.split(' ')
        };
      } else {
        return findCites(el.parentElement)
      }
    } else {
      return undefined;
    }
  };
  var bibliorefs = window.document.querySelectorAll('a[role="doc-biblioref"]');
  for (var i=0; i<bibliorefs.length; i++) {
    const ref = bibliorefs[i];
    const citeInfo = findCites(ref);
    if (citeInfo) {
      tippyHover(citeInfo.el, function() {
        var popup = window.document.createElement('div');
        citeInfo.cites.forEach(function(cite) {
          var citeDiv = window.document.createElement('div');
          citeDiv.classList.add('hanging-indent');
          citeDiv.classList.add('csl-entry');
          var biblioDiv = window.document.getElementById('ref-' + cite);
          if (biblioDiv) {
            citeDiv.innerHTML = biblioDiv.innerHTML;
          }
          popup.appendChild(citeDiv);
        });
        return popup.innerHTML;
      });
    }
  }
});
</script>
</div> <!-- /content -->




</body></html>
