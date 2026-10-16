# longcat-avatar

Avatar falante local do INEMA: **foto + áudio → vídeo com boca sincronizada**, rodando no nosso servidor (NVIDIA GB10, ARM) com o [LongCat-Video-Avatar 1.5](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.5) (Meituan, MIT).

Alternativa local ao estúdio HeyGen: sem crédito, sem API, sem fila.

> **Status:** piloto em andamento (16/10/2026). Ainda não medido: tempo por segundo de vídeo, memória de pico e qualidade comparada ao HeyGen.

## O que este repo traz

O código do modelo continua no espelho [`inematds/LongCat-Video`](https://github.com/inematds/LongCat-Video). Aqui fica só a nossa camada:

| Arquivo | Para quê |
|---|---|
| `scripts/baixar_pesos.py` | Baixa só o que o Avatar 1.5 int8 usa (~43 GB em vez de ~158 GB) para `/mnt/hd8t/longcat-weights`. Retoma se cair. |
| `scripts/setup_venv.sh` | Cria `.venv` reaproveitando o PyTorch cu130 do sistema e põe `compat/` no caminho. |
| `compat/flash_attn*` | Substituto do flash-attn 2/3 via `scaled_dot_product_attention`. O flash-attn 2.7.4 do upstream não tem pacote para ARM + Blackwell. |
| `scripts/avatar.sh` | Roda a geração em 1 GPU, com teto de memória (`systemd-run`, padrão 90 GB) e log em `saidas/<nome>/`. |

## Uso

```bash
python3 scripts/baixar_pesos.py        # uma vez (~43 GB)
./scripts/setup_venv.sh                # uma vez
./scripts/avatar.sh entradas/exemplo-upstream/entrada.json          # 1 segmento = 3,7 s, 480p
./scripts/avatar.sh minha/entrada.json 5 480p meu-teste             # 5 segmentos ≈ 16 s
```

Formato da entrada (caminhos relativos ao json):

```json
{
  "prompt": "descrição da cena em inglês",
  "cond_image": "foto.png",
  "cond_audio": {"person1": "fala.mp3"}
}
```

Variáveis: `LONGCAT_DIR` (espelho do código, padrão `~/projetos/LongCat-Video`), `LONGCAT_PESOS` (padrão `/mnt/hd8t/longcat-weights`), `LONGCAT_MEMORIA` (padrão `90G`).

## Pesos usados

| Origem | Partes | Tamanho |
|---|---|---|
| LongCat-Video (base) | tokenizer, text_encoder (UMT5), vae, scheduler | ~23 GB |
| LongCat-Video-Avatar-1.5 | base_model_int8, lora (distill), whisper-large-v3, vocal_separator | ~22 GB |

O DiT base de 54 GB e o Avatar em bf16 (32 GB) não são baixados.

## Próximos passos

1. Rodar o exemplo do upstream e medir tempo e memória.
2. Rodar com foto e voz do Nei e comparar com o HeyGen.
3. Se valer: entrada simples (foto + mp3 + texto PT), vídeos longos por segmentos, guia PT/EN/ES.

Licença: MIT (a mesma do LongCat-Video).
