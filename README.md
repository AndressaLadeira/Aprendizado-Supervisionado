## Trabalho de Regressão: Previsão do valor de apartmentos em Minas Gerais utilizando o pacote 'tidymodels'

Carregando pacotes.

```{r}
#| message: false
#| warning: false
library(tidymodels)
library(modelsummary)
library(finetune)
library(dplyr)
library(baguette)
library(readxl)
library(glmnet)
library(ranger)
```

Leitura de dados.

```{r}
dados <- read_excel("C:/Users/andre/Downloads/Trab. Aprendizado Supervisionado.xlsx") |>
  mutate(across(where(is.character), as.factor),  # Converte variáveis categóricas para fatores
  Valor = as.numeric(Valor))  
dados <- na.omit(dados)
```

Verificando a estrutura dos dados.

```{r}
dados <- dados |> 
  mutate(across(where(is.character), as.factor))  # Convertendo variáveis categóricas em fatores

dados |> glimpse()
```

Estatística descritiva.

```{r}
datasummary_skim(dados)
```
Separando dados de treino e teste, com 75% dos dados para treino.

```{r}
set.seed(16)
dados_split <- initial_split(dados, 
                             prop = 0.75,
                             strata = Cidade)  # Supondo que a cidade seja relevante para estratificação

dados_train <- training(dados_split)
dados_test  <- testing(dados_split)
```

Definindo a receita com normalização das variáveis numéricas e dummificação das variáveis categóricas.

```{r}
normalized_rec <- 
  recipe(Valor ~ ., data = dados_train) |> 
  step_normalize(all_numeric_predictors()) |>
  step_dummy(all_nominal_predictors())
```

Definindo os modelos de regressão.

```{r}
linear_reg_spec <- 
  linear_reg(penalty = tune(), mixture = tune()) |> 
  set_engine("glmnet")

tree_spec <- decision_tree(tree_depth = tune(), min_n = tune(), cost_complexity = tune()) |> 
  set_engine("rpart") |> 
  set_mode("regression")

rforest_spec <- rand_forest(mtry = tune(), min_n = tune(), trees = tune()) |> 
  set_engine("ranger") |> 
  set_mode("regression")
```

Definindo o workflow.

```{r}
normalized <- 
  workflow_set(
    preproc = list(normalized = normalized_rec), 
    models = list(linear_reg = linear_reg_spec,
                  tree = tree_spec,
                  rforest = rforest_spec)
  )
```

Simplificando os nomes dos modelos.

```{r}
all_workflows <- 
  bind_rows(normalized) |> 
  mutate(wflow_id = gsub("(normalized_)", "", wflow_id))
```

Realizando grid search e validação cruzada.

```{r}
race_ctrl <- control_race(
  save_pred = TRUE,
  parallel_over = "everything",
  save_workflow = TRUE
)

set.seed(1503)
race_results <- all_workflows |>
  workflow_map(
    "tune_race_anova",
    seed = 1503,
    resamples = vfold_cv(v = 10, dados_train, repeats = 2),
    grid = 25,
    control = race_ctrl
  )
```
Extraindo métricas de desempenho.

```{r}
collect_metrics(race_results) |> 
  filter(.metric == "rmse") |>
  arrange(mean)
```
Visualizando o desempenho dos modelos.

```{r}
IC_rmse <- collect_metrics(race_results) |> 
  filter(.metric == "rmse") |> 
  group_by(wflow_id) |>
  arrange(mean) |> 
  ungroup()

ggplot(IC_rmse, aes(x = factor(wflow_id, levels = unique(wflow_id)), y = mean)) +
  geom_point(stat="identity", aes(color = wflow_id), pch = 1) +
  geom_errorbar(stat="identity", aes(color = wflow_id, 
                                     ymin=mean-1.96*std_err,
                                     ymax=mean+1.96*std_err), width=.2) + 
  labs(y = "", x = "method") + theme_bw() +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
```
