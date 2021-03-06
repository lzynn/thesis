install.packages("glmnet")
install.packages("flare")
install.packages("gam")
install.packages("randomForest")
install.packages("foreach")
install.packages("doParallel")

library(devtools)
install_github("vqv/ggbiplot")

setwd("E:/master/OnlineNewsPopularity")
#setwd("/Volumes/CLASS/master/OnlineNewsPopularity")
data_orginal <- read.csv("OnlineNewsPopularity.csv") #url, timedelta are non-predictive
summary(data_orginal) #something data point are wrong


sum(data_orginal$n_unique_tokens > 1)
which.max(data_orginal$n_unique_tokens) # data$n_unique_tokens[31038] = 701, we can't get a rate like that. So as this point other two rates, remove!
data_orginal <- data_orginal[-31038, ]

set.seed(200)
index <- sample(1:39643)
test <- data_orginal[index[27751:39643],]
data <- data_orginal[index[1:27750],]

no_norm <- data[, c(2, 14:19, 32:39)]
#no_norm[, -1] <- lapply(no_norm[, -1], factor)
be_norm <- data[, -c(1, 2, 14:19, 32:39)]
be_norm$shares <- log(be_norm$shares)
mean_norm <- colMeans(be_norm)
sd_norm <- sqrt(colSums((t(t(be_norm) - mean_norm))^2)/39642)
after_norm <- t((t(be_norm)-mean_norm)/sd_norm)

data1 <- cbind.data.frame(no_norm, after_norm)

no_test <- test[, c(2, 14:19, 32:39)]
be_test <- test[, -c(1, 2, 14:19, 32:39)]
be_test$shares <- log(be_test$shares)
after_test <- t((t(be_test)-mean_norm)/sd_norm)

test1 <- cbind.data.frame(no_test, after_test)


# linear
library(faraway)

linearm <- lm(shares ~ ., data = data1)
step(linearm)
step(linearm, k = log(39643))

data2 <- data1[, c(3:6,8:12,16:17,21:23,25:30,32:35,38:39,41,44,51,56:58,60)]

cv_linear <- function(data, form, cv_num, seed = 250)
{
  set.seed(seed)
  n <- dim(data)[1]
  cv_cut <- sample(1:n)
  errors <- rep(0, cv_num)
  for(i in 1:cv_num)
  {
    test <- data[cv_cut[round((i - 1)/cv_num*n + 1):round(i/cv_num*n)],]
    train <- data[-cv_cut[round((i - 1)/cv_num*n + 1):round(i/cv_num*n)],]
    cv_lm <- lm(form, data = train)
    pred <- predict(cv_lm, test)
    errors[i] <- mean((test$shares - pred)^2)
  }
  return(mean(errors))
}

cv_linear(data = data2, form = shares ~ ., cv_num = 10) # error: 0.8767219
cv_linear(data = data1, form = shares ~ ., cv_num = 10) # error: 1.257342
linear_model <- lm(shares ~ ., data = data1)
plot(linear_model) # it seems like not fit the linear assumption, because of QQ plot
vif(linear_model) # title_subjectivity, title_sentiment_polarity  > 10

a <- vif(linear_model) 
a[which(a>10)]

# lasso(glmnet)
matrix_data <- as.matrix(data2)
library(glmnet)
cv_lasso <- cv.glmnet(matrix_data[, -33], matrix_data[, 33], alpha = 1)
plot(cv_lasso)
mean(cv_lasso$cvm) #0.9016617

matrix_data <- as.matrix(data1)
cv_lasso <- cv.glmnet(matrix_data[, -60], matrix_data[, 60], alpha = 1, nfolds = 10)
plot(cv_lasso)
mean(cv_lasso$cvm) #1.287204

# lasso(flare)
library(flare)
cv_sqlasso <- slim(matrix_data[, -12], matrix_data[, 12], method = "lq", nlambda = 40, lambda.min.value = sqrt(log(11)/39643), q = 2)

cv_sqlasso <- function(data, cv_num, method, q, seed = 250)
{
  set.seed(seed)
  n <- dim(data)[1]
  cv_cut <- sample(1:n)
  errors <- rep(0, cv_num)
  for(i in 1:cv_num)
  {
    test <- as.matrix(data[cv_cut[round((i - 1)/cv_num*n + 1):round(i/cv_num*n)],])
    train <- as.matrix(data[-cv_cut[round((i - 1)/cv_num*n + 1):round(i/cv_num*n)],])
    cv_lasso <- slim(train[, -12], train[, 12], method = method, nlambda = 40, lambda.min.value = sqrt(log(11)/39643), q = q)
    pred <- predict(cv_lasso, newdata = test[, -12], lambda.idx = c(40:40), Y.pred.idx = c(1:1))
    pred <- as.data.frame(pred)
    errors[i] <- mean((test[, 12] - pred[, 1])^2)
  }
  return(mean(errors))
}
lq_cv <- cv_sqlasso(data = data2, "lq", 2, cv_num = 10) # error: 0.9823111
lad_cv <- cv_sqlasso(data = data2, "lq", 1, cv_num = 10) # error: 0.983656

# GAM
library(gam)
cv_gam <- function(data, form, cv_num, seed = 250)
{
  set.seed(seed)
  n <- dim(data)[1]
  cv_cut <- sample(1:n)
  errors <- rep(0, cv_num)
  for(i in 1:cv_num)
  {
    test <- data[cv_cut[rouPK     ! �,lok     [Content_Types].xml �(�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 ���N�0��H�C�+j�q@��a�&1 $�-M�8���fcB��B�Q����v&�]c�-�Ζl\�XV:���d�����e�U�8%�����j��{���-����?p���F`�<Xک\hD�װ�^ȵX���t6��yl5�t��ؘ�=���$�A����WɄ�FK��o���
�L1Xk�7��x�C���1�J��l!B|a��.�?�[�E:(]Ui	��MC(�
k�ؘ"�E#���>㟂��e<0H{�$���O���L�!ƽ��I�Ϲ�[4�������yM-2pN����o�y�	���m�sOB��Ӑv5�ɑ���C{�(P�<�g�/   �� PK     ! �U0#�   L   _rels/.rels �(�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 ��MO�0��H�����ݐBKwAH�!T~�I����$ݿ'T�G�~����<���!��4��;#�w����qu*&r�Fq���v�����GJy(v��*����K��#F��D��.W	��=��Z�MY�b���BS�����7��ϛז��?�9L�ҙ�sbgٮ|�l!��USh9i�b�r:"y_dl��D���|-N��R"4�2�G�%��Z�4�˝y�7	ë��ɂ���  �� PK     ! ���  ?   xl/_rels/workbook.xml.rels �(�                                                                                                                                                                                                                                                                 ���j�0E����Ѿq2}P�qf�R�m�~�p�8Lb[}��kR:���n�1H�����w��$gdI
��veck����#��hKl�%���W�Wj��`�>��b���o��P�!q=�8���c�k٣>bMr����5 ?��R�?�� �����k��j4=;�ё�2���D��&V�['��e�͚��B��X��͖�5��?C�ǩ�8Y��_Fc��6v�9��.r�j(z*����ϳ1o��ȳ��?   �� PK     ! 5�@�"  s     xl/workbook.xml�TM��0�#�,���i�m�&�m�J���{��:Ncձ���V���$!P�e\2��y�=�}��
�RL&F\1]�O���o��uTTj�S|��f�_-��vZ (��ʹ&�}�*^S;�W�RjSSC��mc8-lŹ���̯�Px@H�K0tY
�7��5Wn 1\R��J4vD��K�jjm�1]7 �R�s�Q͒��҆�$�>�xD��
��h�K7((�/	|B�ٲ�?�#�4�i��"1�Ժ��)��P�o�mV���J�(�����������n&��q�t;�ԝt�(��Z+�P�_��וw�#��
á):ٲ%|)K��>PW����ɧ=�Y�(e5����<�e׭O��/ź!:=U�c�������~��R<H -0̽�b_9��q���78��G�zc?tMO�%u��#�H�}�;�����B�q.ȴ#�O�u}E���(��	���؋�ЛG��[G�0�o�M����߶+���wUVԸ��� ��G^���6A���X�?�ʾ  �� PK     ! ��h�  E     xl/sharedStrings.xmlt��N�0��H���;��0!�v�@{� ��Zk���.�{z� �����V��W�I�r��=.�H�԰��۽o���4�P�FR����
U�\+Z������E�E:�d�O}D�Ǿ�z�	�,��\�|Du�r��� |��T�rUXx��ȅ���W�Qt��(�iF/�M|�|&�S;��e���l!JC����sLhj�H00����A�P^��ZԩmR�KC?+%#�`}�O"�9�ܳv�%�&�� ��_C��A�  �� PK     ! u>�i�  �     xl/theme/theme1.xml�Y[��F~/�?�;�I���l�N��&!��ql��Ɏ4F3ލ	��<��PHK_
}�C)4�З����6�=3����8�˦�%kX��w�|s��7]�t/��N9aIۭ^��N�lB�Y۽5����J&����%��ݏ?��vD�c�}�wPۍ����|͈_`s���)Kc$�6��'):�1-�*��#��N�bp{}:%c��Kww�O�6\6�iz ]c�Ba'�U��K��9B��B?v<����P�<h���w/��NnD�[�n��r��`rXS}��ѺS�󽠳�� Tl���~��� ��0Ҍ���ﶺ=?�j������ի^�_�������+P����!D��+P��-1i�B��+P�6��J��5�E�$�����hא)�W�����y��jXW��b����bt�� H E�$�X�����CD�(%��EPxs�0͕ZeP����ԕ���H����	�h�|>N�\��O���A�?{v�����_O=:y�s޷re�]A�L�{��W}����/߿|�u��i<��/~���o���=����o��x����_���c��N�F:|Hb̝k�ع�b��?�of1�1,P�-��"2�ז��p]l��v
*c^^�5�D�BK�W�� �3F�,���K��p��읧w�#[�!J��s�WbsFؠy��D�N�p�3v��etw1�O�)�l*�;��"bɐ��B*�����T�ٿ�t�����L$��Z�15�x-�m.�(�z����l$��X����L�0eN�9��\Oa�Zү���ӾO���L9���C���;#ϭ�I��O�!�(rn0a��3�����lM�m��t�-�@\uJE��'�Ԓ�˘���NV*�oHzL�3���������5�4���]Լ��;u唆o������/����A�?����m����u�� ��Z]����)��@,)��j��a^��Qm*��r���Gp�o�,E��I����� BsX�W�6t�s�3���u�jVb|ʷ�=,�}6���ժܛf���(�+���"C�b�v�v�3�W^��oBB��$Q��h�!�"�Fv.,ZM�~��Uס j���Ɂ�V���� �T���Sv$�ʮLιfz[0�^��XU@���uxrtY��F�Z��$�2���թ��g�[EJz2�����h��\K9�4ѕ�&�q��>���Ѽ�Na���j��/�38<�4{��FY�)=ģ,�Jt25����CI�v����@�!�[���%�Y������I��)=�Z��tv
�i���2{��dH�A49vFt��DPb~�*8!��Y4'�3�BV�ߩ�)�]�@Q�P֎�<B����yW"�����1���1C@7C8��	��gݳ�j9M4�9�P9k����M��b5Xeҭ��к�J�P�����kL��3��d�)�R��V��9.�H[⶞#��xۙ�NW�� V�JU��Ç�m���x��xAW��/)�E_v����"�D�F�+g���{��w��懥J��W)5�N����z��W+�n�L,"��~��e Qt�zQ��_��Yۅ1��L}^)+���K����C@t��A����V�3(y�n��
�n���ޠ������)�ש�^�o��j���"�7[��W�u�F���:�e�<��<^�k�o   �� PK     ! ���m�       xl/styles.xml�T[k�0~�?����8K���45�1h{Ul9��HJ�l��ȗĥc�u���w�sSz�J����\�G!FL���j��E���:�**�b>2�����R뎂��s ���޹fE�-�LR{���O���T�#�1�V�_���a8'�r�{��,�DR�xh�Rˆ:�傻c���,W�;��
��F3Z�6��x�Й^��4���] (�u�K���,	-�H �:�(!a�'���V΢R����'�zT��*�/o����@OT�%�$OK-�A��vE%�=���[ý[M%��{Cן�Or��7�c8,\�B�XŞ �
�Q(h���W0=L���(N&H0O��T0��z��<�v@���ޟN7��j�eyZq�ӊ
�Jr ��	q���[����:�B��*�0����b��+��c���s|@��~F��~g��_0�3@����f՞K�8�,]qNQ���A������O��	K5x}�O�u>�w�S���`���0^p����y�����q�׋`vɒ`��7A2�^o6�2���_��}��v/L��b����6C����-�����(Оr_���c�AqF�lN�b~�Eś�l}�Ʉ{��W"$Q4�m���Lp5�j���
M�/I����|�  �� PK     ! �+�":  G     xl/worksheets/sheet1.xml�W˒�6ݧ*�@�J�ˀm�f�1��b�R�L�����D@���>W`c�CW���}H�\!��}y�3��U��b/�&K���)-�{�ǟ��F�j���H���;��/��?���z�/S	2�^�PZ:�Z���Z!%.���*G~Vg�.+�NmP�����j��B�28Ւ$I��$~�qA�$���ח�����xI�U/��SL�R�,��mRY�c�� :f��M7Q|������Ӹ"5I���n�S�[u�B&wwJ��]�p���u'�uYuw�@��Z�K�����N���?����Aʺu`)QL�p���@7������ﰄگ1�~_/lk�{%�p�^3��������h��pN�>�c�,���R�_)O�a�[���D/{�6Ӱ������$������H�	��7��,r}����HC�X�io��5�nӭ>���U�*�Q���%b[w ��4�e�8�5��q��ڀ���v����ڌ�-�Vc[8��c[4�Y�MZ=7v�r{��f��g �5_P�Orw�=��V�Dڞʧ�?��r0���u�&L�'�׈�i\��}1��PGx$��b)�����X�Rt�� �6O�ya���p(��	<"l~�08���a�v��Nd�յ'�e��v��wf�p����2<��k�=�-G�w��t��X�u����DV�Ȗ/?KԸ�G돧��]1��Dx�����ν���٢���p���bq4,�q-e�߂s�`��"�aDn�����A�۴�������3�Ƶ9�Á�p4�G�Y��������N��Êk��~�wŁE7��h�ȍ�G�en�2�p�[��n#E��Q��r�qo���������w���L��!�3�s�H`��aFX~���C �Y���u�-ڽ6�O�)�7'��y�x0��3x4���?�>C;?��m�+{���5L�ֿ.���R�:�I�x;7�ohun������T�?e�lL�v6 �6Dh������h�n�\$8%�ݎ#�zuN�З茿�����X�j���FS�5OJ6ư�H(�%�_X1t���_B��`CT?��  �� PK     ! o�llH  m   docProps/core.xml �(�                                                                                                                                                                                                                                                                 ��QO�0��M����NM,Q�'���E�[��"-M[e���3|lϹ_Ϲi�<�&�c�V�(�
@�V�j��m�
�Q`S�5����eq}�qMyk�Ŵ�����,�:G�s�bly���;�w������c���N���1�C=�	)��ԟ� �ch@�r�Q������)kwԾ�)�-�(N'c�uQ�1|����7CհV��8�"�r̵��M�����ԯ�a֭��w5�������C���G�*g�5}|*W�HH|�4LH���&���{��|v����!�%!4Y�8�π"���  �� PK     ! ����   �     xl/calcChain.xmld�Y
�0�w�;�y״�JS��z���m!KI��퍢��@�I~f��>�$w��5�A>΀��ju��z9�V@����hd�D�r8(�b��V������nC�*�ƦC*7c��jk�:��r�W�N�lAU���2��9�64D�O��C�?9&rJ��n�����1,c�:꿬c�����Lb��0KB��i���  �� PK     ! ;=��     docProps/app.xml �(�                                                                                                                                                                                                                                                                 ��Ao�0����9�P��bH7��a��gM�c��$����׏����vڍ||x�DI�:_����P��k��x�}��$
$j�c�JŽ~�NmrL���-QZI������ǁ'M̝!n�^Ʀq�}� ��-�;	�PC}�΁bJ\����u�>�����w��Rw6G�_���b�-ط��K%��Z�a���1Aɋ��K��Q��V=X��@���v+��a��Do�3�k�M�X����K̯�*ɆI˹w^��z9��6�w�<��fc2��x9'&�	g;�Mg���+�Id�c�L8^6��IR�\xŧ�����kQm[���8��,�G^i�CȺ5a����`���W�˻E����iJ^>��  �� PK-      ! �,lok                     [Content_Types].xmlPK-      ! �U0#�   L               �  _rels/.relsPK-      ! ���  ?               �  xl/_rels/workbook.xml.relsPK-      ! 5�@�"  s               	  xl/workbook.xmlPK-      ! ��h�  E               \  xl/sharedStrings.xmlPK-      ! u>�i�  �               �  xl/theme/theme1.xmlPK-      ! ���m�                 f  xl/styles.xmlPK-      ! �+�":  G               '  xl/worksheets/sheet1.xmlPK-      ! o�llH  m               �  docProps/core.xmlPK-      ! ����   �                 xl/calcChain.xmlPK-      ! ;=��                   docProps/app.xmlPK      �  �                                                                                                   ��`1b������Rd��Da�wi�Ag����,�1KafB}'��K�ȧ���&��ZwЯ���B51�Ƽ���ݘ��E� 7	�kd�V&�W�=$�8�O�c��s1*����W�t��0v���ڀX�s�~l5�j�X<jȂ�g�&�. ��4�P�����x�Z��ZÖy��2s 6�Y��B�҃��B��9�����}Բm�O]���/���(����N��MZ[�n���[�q��|"�&�j8u�R�ZZ7���s!��� a��:6#�E�'M\�/���H�U��l+��k�}�{?w������{�$b�1��Š��C�6��c�tӉ���K:�]���af��Ъ��Cqwَ�$jw}�r�Zz�:������oGhew�M��y��eZV�J�������8t�XP�����.G_�NJ���HM�b�Z!Z�����؅�`�}3��=��<v�J�^L-��L<t5/≠ʢ9�N�	�YQt��ŷi�x�\�a��3�Z�ZP����\C����f��8�v��#��������%����o�"=��Ж{�T�<�p��e[t�|�+q�R�H�v�l���0L�Bg3���p��a[����Շ1��	d	سD�Gc������}��	2b#����!?8 ��;ֶߴ�X��w-��>��n:���M�7ˉ|ײ�ȥ�1>�Zq���0J�0k����ީl�Y�^�n�z��`����v5.��J$������P�u�̜04���2bC@&$cٛt((姦.��l?h���C'��ӹ��2n�����=߯��˩�l�a�����g�c�'��̯ ���OC�G.��D�x�C( ��i��$�M��F��CfG�z�MR��I�Hd���ZB�i�װ}��]GO�7�P��>����q@�:�ś|��]�Z6[Nqo`�&�<x�S�i�M�ߩa�r��;No��p�lpjz��ɣ�n��~�ƎoW�P���>37aR���+;�]����/��a��O��{�f�W�4��&�)�Rɝ���g��`��S��q�_�D?Pi�G+�f֊�A&#U��q],�7s˩���/\�w��N�PL~�?nA�����g�Bv�aV�S ��3��m�&�x���Ӂ��Z�߀kƩ�?�11!2<I�ψ��ޓV~Ҹ�[�{LYK[IO��Ch|f��������n��~������x0���w����x�y،>cf]�c��p
�@�fm���-�]�,��n�`���M�����9��Eo�$j�a N��F/ߍY�����:�P�(�G|/�َXX��\Y3���v*<�pe��~sv�к�;U:}ۼ[奔EffA!+7�?"����☪j>l��`���4�%���0�&da�wjX�nj���6����t4�S�s��o�tI@��LV��ν��}wƖ�;z�B�rmx�;��/�HP���9��4s�I�� ~!3���Xu��JTƝ;T��1�f;ۙX��M�D�d��H��`	��0��jY�Icv��q�_CcD�/<���M��q�R䘡�}�ǂ��pu�oȰ��jY/Q�჆��2��S�~ج$���	6�(�D�F1���bj��mk��z�|�R5;��c��Eۗ������ŭgB�x�^u�q����:�W���y�9�x���ҭ9�_�