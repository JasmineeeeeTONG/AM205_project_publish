---
title: Home 
---

## Contents
{:.no_toc}
*  
{: toc}


## Authors
Rui Fang, Jiawen Tong


## Overview 

### Background
Pattern formation in nature is an intriguing question that has attracted extensive research from biologists and applied mathematicians in the past decades. Originated from Turing's seminal work published in 1952, *The Chemical Basis of Morphogenesis*, reaction-diffusion (RD) equations are the most well-known theoretical models for the mechanism behind self-regulated pattern formation. 

Turing's theory proposed a basic idea that spontaneous pattern formation can be sufficiently explained by diffusive and reactive processes of a system of chemical substances. This idea is simple but profound. Over the years, a variety of RD equations have been developed to model patterns in nature, and remarkable similarities between simulation and reality are observed. 

According to Meinhardt and Klingler, pattern formation processes generally involve the competition between short-range autocatalytic and long-range inhibitory reactions. This leads to two common two-component RD models: **1) the activator-inhibitor model** and **2) the activator-substrate model**. Assumed to trigger local pigmentation, the activator chemicals have an autocatalytic (positive) feedback on their own production. The inhibitory process reduces the activator production, either through inhibitor chemicals, which are produced simultaneously with the activator production, or through depletion of substrate chemicals, which are necessary precursors for the activator production.

### Our work
In this project, we studied two representative two-dimensional RD models: the Gierer-Meinhardt model (GM, activator-inhibitor) and the Gray-Scott model (GS, activator-substrate). To simulate the pattern formation process with relatively high accuracy and efficiency, we implemented four different numerical methods for each model([GM](https://jasmineeeeetong.github.io/AM205_17Fall_Project_Publish/GM_Numerical_Methods.html), [GS](https://jasmineeeeetong.github.io/AM205_17Fall_Project_Publish/GS_Numerical_Methods.html)) and compared the results and performance of different methods. We also explored the effects of domain([GM](https://jasmineeeeetong.github.io/AM205_17Fall_Project_Publish/GM_Domain.html), [GS](https://jasmineeeeetong.github.io/AM205_17Fall_Project_Publish/GS_Domain.html)), initial conditions([GM](https://jasmineeeeetong.github.io/AM205_17Fall_Project_Publish/GM_Initial_Conditions.html), [GS](https://jasmineeeeetong.github.io/AM205_17Fall_Project_Publish/GS_Initial_Conditions.html)), and parameter settings([GM](https://jasmineeeeetong.github.io/AM205_17Fall_Project_Publish/GM_Parameters.html), [GS](https://jasmineeeeetong.github.io/AM205_17Fall_Project_Publish/GS_Parameters.html)) on pattern formation by changing one factor at a time while keeping the others fixed.

This website displayed the experiments we did for this project in the following pages. To gain an intuitive sense on the pattern formation process modeled by the RD equations, please go to the animation pages ([GM](https://jasmineeeeetong.github.io/AM205_17Fall_Project_Publish/GM_Pattern_Animation.html), [GS](https://jasmineeeeetong.github.io/AM205_17Fall_Project_Publish/GS_Pattern_Animation.html)). 


## RD Equations

### The Gierer-Meinhardt Model

$$\frac{\partial u}{\partial t} = D_u (\frac{\partial^2 u}{\partial x^2}+\frac{\partial^2 u}{\partial y^2}) + \frac{\rho}{v} (\frac{u^2}{1+\kappa u^2}) - \mu_u u + \rho_u$$

$$\frac{\partial v}{\partial t} = D_v (\frac{\partial^2 v}{\partial x^2}+\frac{\partial^2 v}{\partial y^2}) + \rho (\frac{u^2}{1+\kappa u^2}) - \mu_v v$$

### The Gray-Scott Model

$$\frac{\partial u}{\partial t} = D_u (\frac{\partial^2 u}{\partial x^2}+\frac{\partial^2 u}{\partial y^2}) -uv^2 + f(1-u)$$

$$\frac{\partial v}{\partial t} = D_v (\frac{\partial^2 v}{\partial x^2}+\frac{\partial^2 v}{\partial y^2}) +uv^2 - (k+f)v$$
