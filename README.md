Como rodar:

```nix develop -c .venv/bin/flask --app src.main run --debug```

ou:

```docker compose up```

### Envio de relatos por e-mail

O formulário envia os relatos por SMTP para `marcusvssasis@gmail.com`. Configure
as variáveis abaixo no ambiente ou em um arquivo `.env` na raiz do projeto
(o `.env` não deve ser commitado):

```dotenv
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=seu-endereco@gmail.com
SMTP_PASSWORD=sua-senha-de-aplicativo
SMTP_FROM=seu-endereco@gmail.com
SMTP_USE_TLS=true
```

`SMTP_HOST`, `SMTP_USERNAME` e `SMTP_PASSWORD` são necessários. `SMTP_PORT`
usa `587` por padrão; `SMTP_FROM` é opcional e, se omitido, usa
`SMTP_USERNAME`. `SMTP_USE_TLS` usa `true` por padrão e habilita STARTTLS.
Para contas Gmail, use uma senha de app gerada pela conta, não a senha normal.
Com Docker Compose, as variáveis do `.env` são repassadas ao contêiner; reinicie
o serviço após configurá-las. Sem a configuração SMTP, o formulário informa
quais variáveis faltam e permite tentar novamente quando o serviço estiver
configurado.
