# American calls: To exercise, or not to exercise, that is the question

Consider two vanilla call options on the same non-dividend-paying stock, with the same strike $K$ and expiry $T$, priced under identical market assumptions. They differ in one respect only: the European call can be exercised solely at expiry, while the American call can be exercised at any time up to expiry. Let us price both with a Monte Carlo simulation.

<!-- Monte Carlo pricing demo goes here -->

The two prices almost perfectly coincide, and the small observed difference is merely due to MC simulation errors. This is surprising at first. The American call grants strictly more rights, so we would expect it to carry a premium over the European one. Why does the extra flexibility of early exercise have no value?

The answer is that, for a call on a stock that pays no dividends, with non-negative interest rates, exercising before maturity is never optimal. The holder is always better off selling or keeping the option than exercising it. The American call is therefore only ever exercised at maturity, which makes it equivalent to a European call. The rest of this article shows why.

**Setup.** A little notation before getting into the explanation. $S$ is today's stock price, $K$ the strike, and $\tau = T-t$ the time left until expiry. $r$ is the risk-free interest rate, compounded continuously, so $Ke^{-r\tau}$ is the cash you would set aside today to have exactly $K$ at expiry. Finally, $c$ and $p$ are the prices of the European call and put, and $C$ and $P$ are the prices of the American call and put on the same underlying.

We assume frictionless markets (no costs, and you can borrow and lend at $r$) and no arbitrage, meaning no free lunch: if one portfolio pays at least as much as another in every scenario, it cannot be cheaper today. Throughout, we take $r \ge 0$, and we assume no dividends until Section 2.

## 1. The right question: exercise or sell?

Holding an in-the-money American call, you can exercise it for $S-K$ today, or sell it. The option is only worth exercising if the market price is at most $S-K$. So we need to compare $S-K$ with what the option is worth.

At expiry, the call payoff can be split exactly:

$$
\max(S_T-K,\,0) = (S_T - K) + \max(K-S_T,\,0). \tag{1}
$$

The identity above can be interpreted as follows. The payoff $S_T - K$ is that of a binding commitment to buy the stock at $K$ on date $T$. It is replicated by holding the share and owing $K$, and it is worth $S - Ke^{-r\tau}$ today, since $Ke^{-r\tau}$ is the amount you would borrow now to owe exactly $K$ at $T$. 

A call differs from this commitment in only one way: you may choose not to buy. That freedom matters when $S_T < K$, where the commitment loses $K - S_T$ and the call loses nothing. The call is therefore the commitment plus a payment that offsets its losses and vanishes otherwise. That payment is $\max(K-S_T,0)$, the payoff of a put. The put's price $p$ is thus the market value of the right to decline, which is what we refer to below as the insurance component.

The two sides pay the same amount in every scenario at $T$, so they must cost the same today. If the call were cheaper than the package containing share, debt and put, you could buy the call and sell the package. You would collect the difference now, and the two positions would cancel exactly at $T$, leaving a riskless profit. The reverse trade works if the call were more expensive. Equating the two prices gives put-call parity at any given time to maturity $\tau$:

$$
c = S - Ke^{-r\tau} + p. \tag{2}
$$

Subtracting the exercise value $S-K$ from both sides of Equation (2) gives:

$$
c-(S-K) =
\underbrace{K\left(1-e^{-r\tau}\right)}_{\text{interest on the deferred strike}}
+ \underbrace{p}_{\text{insurance value}}. \tag{3}
$$

Let's take a closer look at the two terms on the right:

- **The interest on the deferred strike** is what you would earn by paying $K$ at $T$ instead of today.
    It is strictly positive whenever the risk-free rate is positive.
- **The insurance value**, the value of being able to walk away from purchasing the stock if it finished below $K$.
    It is also strictly positive, as $p>0$ for a volatile stock.

Both terms are therefore non-negative, and as long as time remains before expiry ($\tau>0$), at least one of them is strictly positive.
Hence, it is true that:

$$
c-(S-K) = \underbrace{K\left(1-e^{-r\tau}\right)}_{\text{interest on the deferred strike}}
 + \underbrace{p}_{\text{insurance value}} \;>\; 0. \tag{4}
$$

And, therefore,

$$
c-(S-K) >\; 0 \Rightarrow  c >\; (S-K) \tag{5}
$$

Since the American call carries every European right plus the right to early exercise, $C \ge c > S-K$. Therefore, selling it always yields more than exercising it. Early exercise is therefore never optimal, the extra right has no value, and $C=c$.

This conclusion rests on the absence of dividends. In the next section we show why this assumption is so important.

## 2. Exception: dividends

The argument so far rested on one assumption: the stock pays no dividends. Once it does,
the early-exercise right of the American call can acquire a strictly positive value. Let us see why.

Consider a stock that pays a continuous dividend yield $q$, so that dividends are a known,
fixed fraction of the price. Equation (1) still holds, since it only concerns payoffs at
maturity $T$. What changes is the cost today of the payoff $S_T - K$. Without dividends,
this payoff is replicated by buying one share and borrowing $Ke^{-r\tau}$. With dividends,
that portfolio also collects every dividend paid before $T$, so it ends up worth more than
$S_T - K$ and no longer replicates it.

<!-- Dividend call pricing demo goes here -->

The solution is to buy fewer shares today and let the reinvested dividends rebuild the position.
Let $N$ be the number of shares held. Over a short interval $dt$, each share pays a dividend
of $qS\,dt$, which buys $q\,dt$ new shares at the price $S$. The price cancels, so reinvesting
every dividend gives

$$
dN = qN\,dt.
$$

In words, the number of shares compounds at rate $q$, exactly like a bank balance earning interest
but counted in shares. Integrating from $t$ to $T$ gives $N(T) = N(t)\,e^{q\tau}$. Starting from
$N(t)=e^{-q\tau}$ shares, we end with $N(T) = 1$ share, worth exactly $S_T$ on every price path.
This costs $Se^{-q\tau}$ today. Combined with the debt $Ke^{-r\tau}$ and the put, no-arbitrage
gives the price of the right-hand side of (1):

$$
c = Se^{-q\tau} - Ke^{-r\tau} + p. \tag{6}
$$

and the counterpart of (3) becomes

$$
c-(S-K) = K\left(1-e^{-r\tau}\right) + p - S\left(1-e^{-q\tau}\right). \tag{7}
$$

Compared with (3), a new negative term $S\left(1-e^{-q\tau}\right)$ appears: the dividends forgone by holding the option
instead of the share.

As $S$ grows, the forgone dividends $S\left(1-e^{-q\tau}\right)$ also grow, while the insurance value $p$ shrinks toward zero, since
a fall below $K$ becomes very unlikely. The interest term $K\left(1-e^{-r\tau}\right)$ remains fixed. For a deep in-the-money call, the right-hand side of
(7) therefore turns negative once $S\left(1-e^{-q\tau}\right) > K\left(1-e^{-r\tau}\right) + p$. In that region, $c < S-K$. Since no-arbitrage requires the American call to be worth at least its exercise value, $C \ge S-K$, and therefore $C > c$.

With *discrete* cash dividends the same goal is reached differently: hold one share and borrow
the present value of the dividends, so that each dividend, when paid, repays its own loan.
Replicating $S_T$ then costs $S$ minus the present value of the dividends, instead of $Se^{-q\tau}$.
In this setting, early exercise can only be optimal immediately before an ex-dividend date.
Between dividends, the no-dividend argument applies, so waiting is better.

## 3. But how about puts?

Puts do not present the same artefacts as calls. It is easy to see why by rearranging Equation (2):

<!-- Put pricing demo goes here -->

$$
p-(K-S) = c - K\left(1-e^{-r\tau}\right). \tag{8}
$$

Now the interest term works *against* holding. Exercising a put gives you $K$
in cash immediately, and that cash earns interest. Holding keeps only the
option value $c$ of the stock moving further. The put's gain is capped at $K$
because the stock cannot fall below zero, and the deeper in-the-money you are,
the less is left to gain. You are better off exercising the right and investing the money
than waiting any longer, once the put is deep enough in the money.

As $S$ falls, the call value $c \to 0$, and the right-hand side turns negative.
The European put then trades *below* its exercise value $K-S$. The American
put can always capture $K-S$ by exercising, so it is strictly more valuable
than the European put whenever $r>0$. At $r=0$, (8) gives
$p-(K-S)=c>0$, hence $p > K-S$. Since the American put is worth at least $p$,
early exercise of a put on a non-dividend stock is not optimal either.