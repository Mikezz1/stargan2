# Implementation of StarGan2 paper

- Разобраться с нормализацией картинок
- Когда все заработает - отрефакторить код модели))))))
- Разобраться с архитектруой сетки
- структурировать трейн луп 
- добавить фид
- reg loss check explosions
- detach fake2 in style div loss
- fix label calc for multidomain setting
- R1 regularization
- style div loss decay
- gradient clipping мб
- MA for training
- noisy labels


- В текущем сетапе style rec и cycle loss сходятся, модель обучается только за счет них (скорее даже за счет первого). Adv лосс не падает вообще, проблема либо в маппинге либо в дискриминаторе? либо в adain 


- Повыкидывать лоссы из ref части - чекнуть какие негативно влияют на сходимость
- найти баг в основной части пайплайна...
- мб все же архитектутра подводит
