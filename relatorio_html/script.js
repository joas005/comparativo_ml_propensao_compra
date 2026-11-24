const METRICAS = [['auc','AUC-ROC'],['f1','F1'],['prec','Precisão'],['rec','Recall'],['acc','Acurácia']];
const REVENUE = {labels:['Não compra','Compra'], qtd:[10422,1908], pct:[84.53,15.47]};

const MODELOS = {
  lr:{nome:'Regressão Logística', arq:'regressao_logistica',
    cv:{auc:.9056,f1:.6212,prec:.5245,rec:.7621,acc:.8559}, dp:{auc:.0065,f1:.0202,prec:.0248,rec:.0102,acc:.0106},
    teste:{auc:.8949,f1:.5962,prec:.5,rec:.7382,acc:.8451}, cm:[[1802,282],[100,282]],
    params:'C = 0,1', tempo:5.10, comb:3, tfit:.18, pre:'Mediana nas numéricas + padronização (StandardScaler); moda + one-hot nas categóricas', fixos:'class_weight = balanced; max_iter = 1000', grade:'C ∈ {0,1; 1; 10}', dif:.0106,
    interp:'Maior interpretabilidade relativa: coeficientes lineares com sinal.',
    intro:'Modelo linear que serve de referência. Cada variável recebe um coeficiente: valores positivos aumentam a propensão de compra e valores negativos a reduzem.',
    impTipo:'coef', impLabel:'Coeficiente (sinal preservado)',
    imp:[['PageValues',2.2104],['Month_Nov',.7025],['ExitRates',-.6056],['Month_Feb',-.5957],['Month_May',-.5577],['TrafficType_13',-.5166],['Month_Sep',.4737],['Month_Mar',-.4712],['TrafficType_8',.4619],['Browser_12',.437],['Browser_7',-.3338],['TrafficType_3',-.324],['TrafficType_5',.3039],['TrafficType_20',.2918],['Month_Dec',-.2872]],
    leitura:'<em>PageValues</em> tem o maior coeficiente (+2,21), bem acima do segundo colocado, <em>Month_Nov</em> (+0,70). Taxa de saída (<em>ExitRates</em>, −0,61) e os meses de fevereiro, maio e março reduzem a propensão. A regularização mais forte testada (C = 0,1) foi a melhor, indicando que o modelo se beneficia de coeficientes mais contidos.',
    grid:[['C=0,1',.9056,.0065,.6212,.7621,.18],['C=1',.9044,.0065,.6169,.7647,.20],['C=10',.9032,.0065,.6159,.7634,.25]],
    gridTit:'Valores de C testados',
    texto:'A acurácia de 84,5% no teste é idêntica à de um classificador que sempre responde “não compra” (2.084 de 2.466 sessões). O modelo identifica 282 dos 382 compradores (recall 73,8%), mas com 282 falsos alarmes (precisão 50,0%). É o de menor AUC-ROC (89,5%) e o que mais se afasta entre validação cruzada e teste (1,1 p.p.).'},
  rf:{nome:'Random Forest', arq:'random_forest',
    cv:{auc:.9285,f1:.6744,prec:.5796,rec:.808,acc:.8791}, dp:{auc:.0037,f1:.0149,prec:.0261,rec:.0142,acc:.0098},
    teste:{auc:.9213,f1:.65,prec:.5558,rec:.7827,acc:.8694}, cm:[[1845,239],[83,299]],
    params:'max_depth = nenhum; min_samples_leaf = 3; n_estimators = 200', tempo:16.38, comb:8, tfit:5.45, pre:'Mediana nas numéricas (sem padronização); moda + one-hot nas categóricas', fixos:'class_weight = balanced; n_jobs = 1', grade:'n_estimators ∈ {100, 200}; max_depth ∈ {nenhum, 12}; min_samples_leaf ∈ {1, 3}', dif:.0072,
    interp:'Relações não lineares; importâncias globais, interpretação menos direta.',
    intro:'Conjunto de muitas árvores de decisão independentes cujas previsões são combinadas por votação. Captura relações não lineares e interações entre variáveis sem exigir transformação prévia.',
    impTipo:'imp', impLabel:'Importância do atributo (%)',
    imp:[['PageValues',.4233],['ExitRates',.0886],['ProductRelated_Duration',.0754],['ProductRelated',.0662],['BounceRates',.0561],['Administrative_Duration',.0394],['Administrative',.0359],['Month_Nov',.0269],['Informational_Duration',.0159],['Informational',.0128],['Month_May',.0115],['TrafficType_2',.0107],['VisitorType_New_Visitor',.0088],['VisitorType_Returning_Visitor',.0079],['Month_Mar',.0078]],
    leitura:'<em>PageValues</em> concentra 42% da importância. Em seguida vêm métricas de navegação (taxa de saída, tempo e quantidade de páginas de produto, taxa de rejeição), o que mostra que o Random Forest se apoia mais no comportamento dentro do site do que no calendário.',
    grid:[['∞ · 1 · 100',.9266,.003,.678,.7163,3.76],['∞ · 1 · 200',.9273,.0032,.6814,.7215,7.36],['∞ · 3 · 100',.9284,.0035,.6739,.81,2.82],['∞ · 3 · 200',.9285,.0037,.6744,.808,5.45],['12 · 1 · 100',.9235,.0047,.667,.8139,2.12],['12 · 1 · 200',.9238,.0046,.6702,.8165,4.06],['12 · 3 · 100',.9261,.0043,.666,.8276,1.94],['12 · 3 · 200',.9267,.0045,.6679,.8316,3.09]],
    gridTit:'profundidade máx. · folhas mín. · nº de árvores',
    texto:'Entre os três, é o que erra menos sessões (acurácia 86,9%) e o mais preciso (55,6%): gera 239 falsos alarmes, contra 284 do XGBoost. Em contrapartida, deixa escapar 83 compradores (recall 78,3%). Folhas mínimas de 3 amostras elevaram o recall na busca em grade (cerca de 81%, contra 72% com 1 amostra), ao custo de menor F1 de validação. Foi o modelo de busca mais lenta (16,4 s).'},
  xgb:{nome:'XGBoost', arq:'xgboost',
    cv:{auc:.9338,f1:.6612,prec:.5403,rec:.8525,acc:.8647}, dp:{auc:.0041,f1:.0147,prec:.0189,rec:.0161,acc:.0086},
    teste:{auc:.9325,f1:.6532,prec:.5321,rec:.8455,acc:.8609}, cm:[[1800,284],[59,323]],
    params:'learning_rate = 0,05; max_depth = 3; n_estimators = 200', tempo:2.07, comb:8, tfit:.81, pre:'Mediana nas numéricas (sem padronização); moda + one-hot nas categóricas', fixos:'objective = binary:logistic; eval_metric = logloss; tree_method = hist; scale_pos_weight = 5,4640 (negativos ÷ positivos em y_treino)', grade:'n_estimators ∈ {100, 200}; max_depth ∈ {3, 5}; learning_rate ∈ {0,05; 0,1}', dif:.0013,
    interp:'Boosting não linear; importâncias globais, interpretação menos direta.',
    intro:'Algoritmo de <em>boosting</em> que constrói árvores rasas em sequência, cada uma corrigindo os erros das anteriores. Foi o de melhor poder de discriminação neste projeto.',
    impTipo:'imp', impLabel:'Importância do atributo (%)',
    imp:[['PageValues',.2979],['Month_Nov',.0806],['Month_Mar',.0722],['Month_May',.0597],['Month_Sep',.057],['ProductRelated',.0224],['Administrative_Duration',.0213],['VisitorType_Returning_Visitor',.0209],['ExitRates',.02],['BounceRates',.0195],['VisitorType_New_Visitor',.0186],['TrafficType_2',.0177],['Administrative',.017],['ProductRelated_Duration',.0169],['Month_Jul',.016]],
    leitura:'Além de <em>PageValues</em> (29,8%), quatro meses (novembro, março, maio e setembro) somam cerca de 27% de importância: o XGBoost explora a sazonalidade mais do que os outros modelos. Os parâmetros escolhidos (árvores de profundidade 3, taxa de aprendizado 0,05) indicam que modelos simples e regularizados bastaram.',
    grid:[['0,05 · 3 · 100',.9337,.004,.6579,.8506,.55],['0,05 · 3 · 200',.9338,.0041,.6612,.8525,.81],['0,05 · 5 · 100',.9337,.0043,.6649,.8362,.94],['0,05 · 5 · 200',.9321,.0045,.6689,.825,.71],['0,1 · 3 · 100',.9335,.0038,.6583,.8506,.2],['0,1 · 3 · 200',.932,.0044,.6586,.8401,.34],['0,1 · 5 · 100',.9319,.0048,.6666,.8204,.32],['0,1 · 5 · 200',.9286,.0044,.6673,.7765,.44]],
    gridTit:'taxa de aprendizado · profundidade · nº de árvores',
    texto:'Detecta 323 dos 382 compradores (recall 84,6%) e deixa escapar apenas 59, o menor número entre os modelos. O preço são 284 falsos alarmes (precisão 53,2%). Tem o maior AUC-ROC, a menor diferença entre validação cruzada e teste (0,1 p.p.) e a busca mais rápida (2,1 s).'}
};
const ORDEM = ['lr','rf','xgb'];
const OUTLIERS = [['Administrative',404,3.28],['Administrative_Duration',1172,9.51],['Informational',2631,21.34],['Informational_Duration',2405,19.51],['ProductRelated',987,8.0],['ProductRelated_Duration',961,7.79],['BounceRates',1551,12.58],['ExitRates',1099,8.91],['PageValues',2730,22.14],['SpecialDay',1251,10.15]];
const CATS = [['Month',10,'May (3.364)'],['OperatingSystems',8,'2 (6.601)'],['Browser',13,'2 (7.961)'],['Region',9,'1 (4.780)'],['TrafficType',20,'2 (3.913)'],['VisitorType',3,'Returning_Visitor (10.551)'],['Weekend',2,'False (9.462)']];
const SPLIT = [['Treino',9864,8338,1526],['Teste',2466,2084,382]];

const f3 = v => v.toLocaleString('pt-BR',{minimumFractionDigits:3,maximumFractionDigits:3});
const f4 = v => v.toLocaleString('pt-BR',{minimumFractionDigits:4,maximumFractionDigits:4});
const pc2 = v => (v*100).toLocaleString('pt-BR',{minimumFractionDigits:2,maximumFractionDigits:2})+'%';
const pp = v => (v*100).toLocaleString('pt-BR',{minimumFractionDigits:1,maximumFractionDigits:1})+' p.p.';
const sg = v => (v>=0?'+':'−')+pp(Math.abs(v));
const pc = v => (v*100).toLocaleString('pt-BR',{minimumFractionDigits:1,maximumFractionDigits:1})+'%';
const n0 = v => v.toLocaleString('pt-BR');
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

function cmGrid(cm){
  const max = Math.max(...cm.flat());
  const cell = (v,ok) => `<div class="v" style="background:color-mix(in srgb,var(--${ok?'accent':'neutral'}) ${Math.round(12+v/max*48)}%,transparent)">${n0(v)}</div>`;
  return `<div class="cm" role="img" aria-label="Matriz de confusão">
    <div></div><div class="h">Previsto: não compra</div><div class="h">Previsto: compra</div>
    <div class="h">Real: não compra</div>${cell(cm[0][0],1)}${cell(cm[0][1],0)}
    <div class="h">Real: compra</div>${cell(cm[1][0],0)}${cell(cm[1][1],1)}</div>`;
}
function buildModel(k){
  const m = MODELOS[k], [[tn,fp],[fn,tp]] = m.cm;
  const linhas = METRICAS.map(([c,r]) => `<tr><td>${r}</td><td>${pc(m.cv[c])} ± ${pp(m.dp[c])}</td><td class="key">${pc(m.teste[c])}</td><td>${sg(m.teste[c]-m.cv[c])}</td></tr>`).join('');
  const melhor = Math.max(...m.grid.map(g=>g[1]));
  const gl = m.grid.map(g => `<tr class="${g[1]===melhor?'best':''}"><td>${g[0]}</td><td>${pc2(g[1])}</td><td>${pp(g[2])}</td><td>${pc2(g[3])}</td><td>${pc2(g[4])}</td><td>${g[5].toLocaleString('pt-BR',{maximumFractionDigits:2})}</td></tr>`).join('');
  document.getElementById('panel-'+k).innerHTML = `
    <h2>${m.nome}</h2>
    <p class="lead">${m.intro}</p>
    <div class="destaque"><strong>Resultado principal</strong><p>No teste, ${m.nome} obteve AUC-ROC de ${pc(m.teste.auc)}, F1 de ${pc(m.teste.f1)} e recall de ${pc(m.teste.rec)}: identificou ${tp} dos ${tp+fn} compradores, com ${fp} falsos alarmes entre ${tn+fp} não compradores.</p></div>
    <p>${m.texto}</p>
    <h3>Configuração técnica</h3>
    <div class="table-wrap"><table class="spec"><tbody>
      <tr><td>Pré-processamento (no pipeline)</td><td>${m.pre}</td></tr>
      <tr><td>Parâmetros fixos</td><td>${m.fixos}</td></tr>
      <tr><td>Grade de busca</td><td>${m.grade}</td></tr>
      <tr><td>Ajustes na validação cruzada</td><td>${m.comb} combinações × 5 folds = ${m.comb*5} ajustes</td></tr>
      <tr><td>Melhor configuração (1º em AUC-ROC)</td><td><strong>${m.params}</strong></td></tr>
      <tr><td>Tempo</td><td>Busca completa (inclui reajuste final): ${m.tempo.toLocaleString('pt-BR',{maximumFractionDigits:1})} s; ajuste médio por fold da melhor configuração: ${m.tfit.toLocaleString('pt-BR',{maximumFractionDigits:2})} s</td></tr>
    </tbody></table></div>
    <h3>Desempenho: validação cruzada e teste</h3>
    <div class="grid two">
      <div class="table-wrap"><table><thead><tr><th>Métrica</th><th>Validação cruzada (média ± dp)</th><th class="key">Teste</th><th>Teste − CV</th></tr></thead><tbody>${linhas}</tbody></table>
        <p class="note">Hiperparâmetros escolhidos: <strong>${m.params}</strong>. ${m.interp}</p></div>
      <figure class="fig"><div class="chart-box"><canvas id="c-${k}-met" role="img" aria-label="Métricas de validação cruzada e teste de ${m.nome}"></canvas></div></figure>
    </div>
    <h3>Matriz de confusão (conjunto de teste)</h3>
    <div class="grid two">
      <div>${cmGrid(m.cm)}
        <p style="margin-top:1rem">Dos ${tp+fn} compradores reais, ${tp} foram identificados (${pc(tp/(tp+fn))}). Dos ${tn+fp} não compradores, ${fp} foram classificados como compradores (${pc(fp/(tn+fp))}), e a especificidade é de ${pc(tn/(tn+fp))}.</p>
        <div class="table-wrap"><table><thead><tr><th>Métrica derivada da matriz</th><th>Valor</th></tr></thead><tbody>
          <tr><td>Especificidade (VN ÷ (VN + FP))</td><td>${pc(tn/(tn+fp))}</td></tr>
          <tr><td>Taxa de falsos positivos</td><td>${pc(fp/(tn+fp))}</td></tr>
          <tr><td>Taxa de falsos negativos</td><td>${pc(fn/(tp+fn))}</td></tr>
          <tr><td>Valor preditivo negativo (VN ÷ (VN + FN))</td><td>${pc(tn/(tn+fn))}</td></tr>
          <tr><td>Acurácia balanceada (média de recall e especificidade)</td><td>${pc((tp/(tp+fn)+tn/(tn+fp))/2)}</td></tr>
          <tr><td>Sessões sinalizadas como compra</td><td>${pc((tp+fp)/(tn+fp+fn+tp))}</td></tr>
        </tbody></table></div></div>
      <figure class="fig"><img src="../output/matriz_confusao_${m.arq}.png" alt="Matriz de confusão do modelo ${m.nome}" loading="lazy"></figure>
    </div>
    <h3>Variáveis mais influentes</h3>
    <p>${m.leitura}</p>
    <div class="grid two">
      <figure class="fig"><div class="chart-box tall"><canvas id="c-${k}-imp" role="img" aria-label="Quinze variáveis mais influentes de ${m.nome}"></canvas></div>
        <figcaption>${m.impTipo==='coef'?'Coeficientes em escala logarítmica (não são %). Barras em destaque aumentam a propensão de compra; barras neutras a reduzem.':'Importância global: indica quanto cada variável contribui, não o sentido do efeito.'}</figcaption></figure>
      <figure class="fig"><img src="../output/importancia_variaveis_${m.arq}.png" alt="Gráfico original de importância de variáveis de ${m.nome}" loading="lazy"></figure>
    </div>
    <h3>Busca em grade (validação cruzada)</h3>
    <p>Cada linha é uma combinação testada (${m.gridTit}). A linha em negrito teve a maior AUC-ROC e foi a escolhida.</p>
    <div class="grid two">
      <div class="table-wrap"><table><thead><tr><th>Configuração</th><th>AUC-ROC</th><th>dp</th><th>F1</th><th>Recall</th><th>Ajuste (s)</th></tr></thead><tbody>${gl}</tbody></table></div>
      <figure class="fig"><div class="chart-box"><canvas id="c-${k}-grid" role="img" aria-label="AUC-ROC por configuração da busca em grade"></canvas></div></figure>
    </div>`;
}
function opts(c,{x,y,horizontal=false,legend=false,min=0,max}={}){
  const num = {title:{display:!!(horizontal?x:y),text:horizontal?x:y,color:c.muted},min,max,grid:{color:c.grid},border:{display:false},ticks:{color:c.muted}};
  const cat = {title:{display:!!(horizontal?y:x),text:horizontal?y:x,color:c.muted},grid:{display:false},ticks:{color:c.muted,autoSkip:false,maxRotation:horizontal?0:45}};
  return {responsive:true,maintainAspectRatio:false,indexAxis:horizontal?'y':'x',
    plugins:{legend:{display:legend,labels:{color:c.text,boxWidth:12}},tooltip:{}},
    scales:horizontal?{x:num,y:cat}:{x:cat,y:num}};
}
const FUNCS = {executivo:[], geral:[], avaliacao:[], lr:[], rf:[], xgb:[], insights:[]};
const live = {}, dirty = {};

FUNCS.executivo.push(c => new Chart('c-executivo',{type:'bar',
  data:{
    labels:['AUC-ROC (Discriminação)','Recall (Captura de Compras)','Precisão (Assertividade)'],
    datasets:[
      {label:'Regressão Logística',data:[MODELOS.lr.teste.auc*100,MODELOS.lr.teste.rec*100,MODELOS.lr.teste.prec*100],backgroundColor:c.neutral2,borderRadius:3},
      {label:'Random Forest',data:[MODELOS.rf.teste.auc*100,MODELOS.rf.teste.rec*100,MODELOS.rf.teste.prec*100],backgroundColor:c.neutral,borderRadius:3},
      {label:'XGBoost (Campeão)',data:[MODELOS.xgb.teste.auc*100,MODELOS.xgb.teste.rec*100,MODELOS.xgb.teste.prec*100],backgroundColor:c.accent,borderRadius:3}
    ]
  },
  options:{...opts(c,{x:'Métrica Estratégica no Teste',y:'Desempenho (%)',legend:true,max:100}),
    plugins:{legend:{display:true,labels:{color:c.text,boxWidth:12}},
      tooltip:{callbacks:{label:i=>`${i.dataset.label}: ${i.raw.toLocaleString('pt-BR',{minimumFractionDigits:1,maximumFractionDigits:1})}%`}}
    }
  }
}));

FUNCS.geral.push(c => new Chart('c-revenue',{type:'bar',
  data:{labels:REVENUE.labels,datasets:[{data:REVENUE.qtd,backgroundColor:[c.neutral,c.accent],borderRadius:3}]},
  options:{...opts(c,{x:'Classe de Revenue',y:'Quantidade de sessões'}),
    plugins:{legend:{display:false},tooltip:{callbacks:{label:i=>`${n0(i.raw)} sessões (${REVENUE.pct[i.dataIndex].toLocaleString('pt-BR',{maximumFractionDigits:1})}%)`}}}}}));
FUNCS.geral.push(c => { const o=[...OUTLIERS].sort((a,b)=>b[2]-a[2]);
  return new Chart('c-outliers',{type:'bar',
    data:{labels:o.map(a=>a[0]),datasets:[{data:o.map(a=>a[2]),backgroundColor:o.map((a,i)=>i===0?c.accent:c.neutral),borderRadius:2}]},
    options:{...opts(c,{x:'% de sessões acima do limite do IQR',y:'Variável numérica',horizontal:true}),
      plugins:{legend:{display:false},tooltip:{callbacks:{label:i=>`${o[i.dataIndex][1].toLocaleString('pt-BR')} sessões (${i.raw.toLocaleString('pt-BR')}%)`}}}}}); });

ORDEM.forEach(k => {
  const m = MODELOS[k];
  FUNCS[k].push(c => new Chart(`c-${k}-met`,{type:'bar',
    data:{labels:METRICAS.map(a=>a[1]),datasets:[
      {label:'Validação cruzada',data:METRICAS.map(([a])=>m.cv[a]*100),backgroundColor:c.neutral,borderRadius:3},
      {label:'Teste',data:METRICAS.map(([a])=>m.teste[a]*100),backgroundColor:c.accent,borderRadius:3}]},
    options:opts(c,{x:'Métrica',y:'Valor (%)',legend:true,max:100})}));
  FUNCS[k].push(c => new Chart(`c-${k}-imp`,{type:'bar',
    data:{labels:m.imp.map(a=>a[0]),datasets:[{data:m.imp.map(a=>m.impTipo==='coef'?a[1]:a[1]*100),
      backgroundColor:m.imp.map(a=>a[1]>=0?c.accent:c.neutral),borderRadius:2}]},
    options:{...opts(c,{x:m.impLabel,y:'Variável',horizontal:true,min:m.impTipo==='coef'?-1:0}),
      scales:{...opts(c,{x:m.impLabel,y:'Variável',horizontal:true,min:m.impTipo==='coef'?undefined:0}).scales}}}));
  FUNCS[k].push(c => { const best=Math.max(...m.grid.map(g=>g[1]));
    return new Chart(`c-${k}-grid`,{type:'bar',
      data:{labels:m.grid.map(g=>g[0]),datasets:[{data:m.grid.map(g=>g[1]*100),backgroundColor:m.grid.map(g=>g[1]===best?c.accent:c.neutral),borderRadius:3}]},
      options:opts(c,{x:'Configuração testada',y:'AUC-ROC médio na validação cruzada (%)',max:100})}); });
});
FUNCS.insights.push(c => new Chart('c-comp',{type:'bar',
  data:{labels:METRICAS.map(a=>a[1]),datasets:[
    {label:'Regressão Logística',data:METRICAS.map(([a])=>MODELOS.lr.teste[a]*100),backgroundColor:c.neutral2,borderRadius:3},
    {label:'Random Forest',data:METRICAS.map(([a])=>MODELOS.rf.teste[a]*100),backgroundColor:c.neutral,borderRadius:3},
    {label:'XGBoost',data:METRICAS.map(([a])=>MODELOS.xgb.teste[a]*100),backgroundColor:c.accent,borderRadius:3}]},
  options:opts(c,{x:'Métrica (conjunto de teste)',y:'Valor (%)',legend:true,max:100})}));
(function(){
  const cab = '<thead><tr><th>Modelo</th>'+METRICAS.map(a=>`<th>${a[1]}</th>`).join('')+'<th>Δ AUC (CV−teste)</th><th>Busca (s)</th></tr></thead>';
  const best = Object.fromEntries(METRICAS.map(([a])=>[a,Math.max(...ORDEM.map(k=>MODELOS[k].teste[a]))]));
  const rows = ORDEM.map(k=>{const m=MODELOS[k];
    return `<tr><td>${m.nome}</td>`+METRICAS.map(([a])=>`<td class="${m.teste[a]===best[a]?'key':''}">${pc(m.teste[a])}</td>`).join('')+`<td>${pp(m.dif)}</td><td>${m.tempo.toLocaleString('pt-BR',{maximumFractionDigits:2})}</td></tr>`}).join('');
  document.getElementById('t-comp').innerHTML = cab+'<tbody>'+rows+'</tbody>';
  const tb = (id,head,rows) => document.getElementById(id).innerHTML = '<thead><tr>'+head.map((t,i)=>`<th>${t}</th>`).join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(x=>`<td>${x}</td>`).join('')+'</tr>').join('')+'</tbody>';
  tb('t-out',['Variável','Outliers (IQR)','% das sessões'],OUTLIERS.map(o=>[o[0],n0(o[1]),o[2].toLocaleString('pt-BR',{minimumFractionDigits:1})+'%']));
  tb('t-cat',['Variável','Categorias distintas','Mais frequente (sessões)'],CATS.map(r=>[r[0],r[1],r[2]]));
  tb('t-split',['Conjunto','Sessões','Não compra','Compra','% de compra'],SPLIT.map(r=>[r[0],n0(r[1]),n0(r[2]),n0(r[3]),pc(r[3]/r[1])]));
  document.getElementById('t-grades').innerHTML='<thead><tr><th>Modelo</th><th>Parâmetros fixos</th><th>Grade de busca</th><th class="num">Combinações</th><th class="num">Ajustes (× 5)</th></tr></thead><tbody>'+ORDEM.map(k=>{const m=MODELOS[k];return `<tr><td>${m.nome}</td><td>${m.fixos}</td><td>${m.grade}</td><td class="num">${m.comb}</td><td class="num">${m.comb*5}</td></tr>`}).join('')+'</tbody>';
  document.getElementById('t-cv').innerHTML='<thead><tr><th>Modelo</th>'+METRICAS.map(a=>`<th>${a[1]}</th>`).join('')+'</tr></thead><tbody>'+ORDEM.map(k=>{const m=MODELOS[k];return `<tr><td>${m.nome}</td>`+METRICAS.map(([a])=>`<td>${pc(m.cv[a])} ± ${pp(m.dp[a])}</td>`).join('')+'</tr>'}).join('')+'</tbody>';
  document.getElementById('facts').innerHTML = [['Sessões analisadas',n0(12330)],['Preditores','17 (10 numéricos + 7 categóricos)'],['Variáveis após one-hot','75'],['Sessões com compra',n0(1908)+' (15,5%)'],['Duplicatas exatas (mantidas)','125 (1,0%)'],['Ausentes e infinitos','0'],['Treino / teste',n0(9864)+' / '+n0(2466)]].map(([a,b])=>`<dt>${a}</dt><dd>${b}</dd>`).join('');
})();


const BRL = v => (v<0?'−':'')+'R$ '+Math.abs(Math.round(v)).toLocaleString('pt-BR');
const BRLs = v => (v<0?'−':'+')+'R$ '+Math.abs(Math.round(v)).toLocaleString('pt-BR');
function fin(){
  const T = parseFloat(document.getElementById('in-ticket').value)||0, C = parseFloat(document.getElementById('in-custo').value)||0;
  return ORDEM.map(k=>{const m=MODELOS[k],[[tn,fp],[fn,tp]]=m.cm,ab=tp+fp;
    return {k,nome:m.nome,tp,fn,fp,ab,T,C,rec:tp*T,perd:fn*T,cst:ab*C,liq:tp*T-ab*C};});
}
function renderFin(){
  const [L,R,X] = fin(), T=X.T, C=X.C;
  const rel = (a,b) => b.liq>0 ? ` (${((a.liq/b.liq-1)*100).toLocaleString('pt-BR',{maximumFractionDigits:1,signDisplay:'always'}).replace('-','−')}%)` : '';
  const dif = (a,b) => a.liq-b.liq;
  const d1 = dif(X,R);
  document.getElementById('bn-hero').innerHTML = `<div class="bn-label">Escolher o <strong>XGBoost</strong> em vez do <strong>Random Forest</strong> representa</div>
    <div class="bn-value">${BRLs(d1)}</div>
    <div class="bn-sub">de resultado estimado no conjunto de teste${rel(X,R)}: ${X.tp} contra ${R.tp} compras identificadas${d1<0?'. Com esse custo por abordagem, o Random Forest rende mais.':'.'}</div>`;
  const card = (t,a,b) => `<div class="bn-card"><div class="bn-label">${t}</div><div class="bn-value">${BRLs(dif(a,b))}</div><div class="bn-sub">${a.tp} contra ${b.tp} compras identificadas${rel(a,b)}</div></div>`;
  document.getElementById('bn-cards').innerHTML = card('XGBoost em vez de Regressão Logística',X,L)+card('Random Forest em vez de Regressão Logística',R,L)+
    `<div class="bn-card"><div class="bn-label">Vendas deixadas escapar (compras não identificadas)</div><ul>${[X,R,L].map(m=>`<li><span>${m.nome}</span><strong>${BRL(m.perd)}</strong></li>`).join('')}</ul></div>`;
  const melhor = Math.max(...[L,R,X].map(m=>m.liq));
  document.getElementById('t-fin').innerHTML = '<thead><tr><th>Modelo</th><th>Compras identificadas</th><th>Abordagens</th><th>Receita capturada</th><th>Custo de abordagem</th><th>Resultado líquido</th></tr></thead><tbody>'+
    [X,R,L].map(m=>`<tr class="${m.liq===melhor?'best':''}"><td>${m.nome}</td><td>${n0(m.tp)} de 382</td><td>${n0(m.ab)}</td><td>${BRL(m.rec)}</td><td>${BRL(m.cst)}</td><td>${BRL(m.liq)}</td></tr>`).join('')+'</tbody>';
  const dc = X.ab-R.ab, dt = X.tp-R.tp;
  const ponto = dc>0&&dt>0&&T>0 ? ` O XGBoost supera o Random Forest enquanto o custo por abordagem for menor que ${BRL(dt*T/dc)} (${pc(dt/dc)} do valor da compra), porque aborda ${dc} sessões a mais para capturar ${dt} compras a mais.` : '';
  document.getElementById('bn-note').innerHTML = `<strong>Como ler.</strong> Os valores são ilustrativos: usam ${BRL(T)} por compra e ${BRL(C)} por abordagem, hipóteses que não constam na base. A receita capturada é o valor das compras que o modelo sinalizou entre as 382 reais do teste, não um ganho incremental, pois supõe que a ação de marketing não altera a conversão. O resultado líquido é a receita capturada menos o custo de abordar todas as sessões sinalizadas, incluindo os falsos alarmes.${ponto}`;
}
FUNCS.insights.push(c => { const f=fin(), v=f.map(m=>m.liq);
  return new Chart('c-fin',{type:'bar',
    data:{labels:f.map(m=>m.nome),datasets:[{data:v,backgroundColor:f.map(m=>m.liq===Math.max(...v)?c.accent:c.neutral),borderRadius:3}]},
    options:{...opts(c,{x:'Modelo',y:'Resultado líquido estimado (R$)',min:Math.min(0,...v)}),
      plugins:{legend:{display:false},tooltip:{callbacks:{label:i=>BRL(i.raw)}}}}}); });
['in-ticket','in-custo'].forEach(id=>document.getElementById(id).addEventListener('input',()=>{renderFin(); if(ativa==='insights') renderCharts('insights');}));

function palette(){return {accent:css('--accent'),neutral:css('--neutral'),neutral2:css('--neutral-2'),text:css('--text'),muted:css('--muted'),grid:css('--grid')};}
let ativa = 'geral';
function renderCharts(id){
  (live[id]||[]).forEach(ch=>ch.destroy());
  const c = palette();
  Chart.defaults.font.family = css('--font-sans'); Chart.defaults.color = c.muted;
  live[id] = FUNCS[id].map(fn=>fn(c)); dirty[id] = false;
}
function showPanel(id){
  ativa = id;
  document.querySelectorAll('.tab').forEach(t=>t.setAttribute('aria-selected', t.dataset.panel===id));
  document.querySelectorAll('.panel').forEach(p=>p.hidden = p.id!=='panel-'+id);
  if(!live[id] || dirty[id]) renderCharts(id);
  history.replaceState(null,'','#'+id);
  window.scrollTo({top:0});
}
function setTheme(t){
  document.documentElement.setAttribute('data-theme',t);
  const b = document.getElementById('themeBtn');
  b.textContent = t==='dark'?'Modo claro':'Modo escuro'; b.setAttribute('aria-pressed',t==='dark');
  try{localStorage.setItem('tema',t)}catch(e){}
  Object.keys(FUNCS).forEach(k=>dirty[k]=true);
  if(live[ativa]) renderCharts(ativa);
}
document.getElementById('themeBtn').addEventListener('click',()=>
  setTheme(document.documentElement.dataset.theme==='dark'?'light':'dark'));
document.querySelectorAll('.tab').forEach(t=>{
  t.addEventListener('click',()=>showPanel(t.dataset.panel));
  t.addEventListener('keydown',e=>{
    const tabs=[...document.querySelectorAll('.tab')],i=tabs.indexOf(t);
    const j = e.key==='ArrowRight'?(i+1)%tabs.length : e.key==='ArrowLeft'?(i-1+tabs.length)%tabs.length : -1;
    if(j>=0){tabs[j].focus();showPanel(tabs[j].dataset.panel);}});
});

ORDEM.forEach(buildModel);
renderFin();
let tema='light'; try{tema=localStorage.getItem('tema')||'light'}catch(e){}
document.documentElement.setAttribute('data-theme',tema);
setTheme(tema);
const inicio = location.hash.slice(1);
showPanel(FUNCS[inicio]?inicio:'executivo');
