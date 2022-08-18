clc
clear all



// lendo a base de dados
base=csvRead('dermatology.data')



// retirando linhas com dados faltantes
base(isnan(sum(base,2)),:) = []



// ordenando as linhas da base (ordem crescente, de cima para baixo) pelo tipo da doença (coluna 35)
[dummy,idx]=gsort(base(:,35),"g","i")
base=base(idx,:);



// vec[i] deve indicar a quantidade de elementos i na coluna do tipo da doença (coluna 35)
// há ao todo 6 classes
vec = []
for i=1:6
    [nb, loc] = members([i], base(:,35), "last")
    vec = [vec nb]
    
end



// obtenção da matriz de rótulos já no formato desejado (em cada coluna, o tipo da doença é indicado pelo valor 1, e os outros pelo valor 0)
Rotulos = zeros(6,358)

for i=1:6
    for k=1:vec(i)
    Rotulos(i,k+sum(vec(1:i-1))) = 1
    end
end



// dados de entrada no formato desejado
E = base(:,1:34)'



// normalização por z-score dos dados de entrada
for i=1:34
    E(i, :) = (E(i, :)-mean(E(i, :)))/stdev(E(i,:)) 
end




//aproximadamente metade das amostras de cada classe será usada para treinamento e o resto para teste
//vec = [111 60 71 48 48 20]
// dessa forma, treino: 56 (1), 30(2), 36(3), 24 (4), 24 (5) e 10 (6) -> totalizando 180 para treino
// teste: 55 (1), 30(2), 35(3), 24 (4), 24 (5) e 10 (6) -> totalizando 178 para teste
//os intervalos dos índices estão mostrados a seguir:

ind_treino = [1:56, 112:141, 172:207, 243:266, 291:314, 339:348]
ind_teste = [57:111, 142:171, 208:242, 267:290, 315: 338, 349:358]



// divisão dos dados de entrada entre treino e teste
E_treino = E(:,ind_treino)
E_teste = E(:,ind_teste)



// divisão dos rótulos entre treino e teste
R_treino = Rotulos(:,ind_treino)
R_teste = Rotulos(:,ind_teste)



[p_treino N_treino] = size(E_treino)



[qtd_out idc] = size(R_treino)



W = ann_FFBP_gd(E_treino,R_treino,[p_treino 121 qtd_out])


// Previsão para os dados de teste
Previsao = ann_FFBP_run(E_teste, W)



[p_teste N_teste] = size(E_teste)



// contador para avaliar a porcentagem de acertos
count = 0
for i=1:N_teste
    
    [a b] = max(R_teste(:,i))
    [c d] = max(Previsao(:,i))
    if b==d
       count = count+1 
    end
    
end



// qtd de acertos (count) dividida pela quantidade de dados de teste (178) e, em seguida, multiplicada por 100
// indica a porcentagem de acertos
disp(100*count/178)












