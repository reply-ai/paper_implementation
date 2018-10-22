---
layout: post
title:  "논문으로 시작하는 딥러닝 - [실습] 이미지 처리 실습하기"
subtitle:   ""
categories: Data
tags: Papers
---


## 들어가기

 딥러닝에 학습을 깊이있게 하고자, 작년부터 논문을 공부하는 것을 시도해봤다. 다만, 많은 경험이 없고 혼자서 하기엔 많은 어려움이 있어
 작년부터 시작된 PR12의 유튜브 동영상을 참고하여 내가 관심 갖는 주제에 대해서는 정리하여 포스팅을 하고자 하였다.
 이번에 선택한 논문은 YouTube에서 사용된 실제 논문으로 추천시스템 공부를 계속 진행해왔기 때문에 제목을 보자 마자 시청한 논문이다.

 * youtube - [youtube 링크](https://www.youtube.com/watch?v=V6zixdCIOqw&feature=youtu.be)
 * slide - [slideshare 링크](https://www.slideshare.net/keunbongkwak/deep-neural-networks-for-youtube-recommendations)
 * blog - [blog 링크](http://keunwoochoi.blogspot.kr/2016/09/deep-neural-networks-for-youtube.html)
 * papers - [papers 링크](https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/45530.pdf)


## 개요

 * Candidate Generation Model
 * Ranking Model
 * A/B Test를 통한 실제 환경 개선

# 실제 상황에서 겪게 되는 이슈들

 * Scale : 엄청난 양의 데이터와 제한된 컴퓨팅 파워
 * Freshness : 새로운 컨텐츠의 빠른 적용
 * Noise : 낮은 meta data 퀄리티, Implicit Feedback 위주 데이터
