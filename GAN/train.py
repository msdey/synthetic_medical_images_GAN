from preprocessing import *
from model import *


import time

X_train = get_images(image_path = x_ray_data)
train_size = X_train.shape[0]

generator_weights = './saved_weights/generator_model.h5'
discriminator_weights = './saved_weights/discriminator_model.h5'

if os.path.exists(generator_weights):
  generator.load_model(generator_weights)
  discriminator.load_model(discriminator_weights)


for step in range(num_steps): 
    tot_step = step
    if step%500==0:
      print("Begin step: ", tot_step)
      begin_time = time.time() 

    # generate real and fake images
    random_index = np.random.randint(0, train_size , size = batch_size)
    real_data_X = X_train[random_index]
    noise = generate_noise(batch_size,noise_shape)
    fake_data_X = generator.predict(noise)

    #concatenate real and fake data samples
    X = np.concatenate([real_data_X,fake_data_X])
    
    if (tot_step % 500) == 0:
        step_num = str(tot_step).zfill(4)
        save_name = os.path.join(img_save_dir + '/' +  str(step_num)+  '.png')
        save_img_batch(fake_data_X, save_name)

    
    
    #add noise to the label inputs
    real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
    fake_data_Y = np.random.random_sample(batch_size)*0.2 
    Y = np.concatenate((real_data_Y,fake_data_Y))
        
    discriminator.trainable = True
    generator.trainable = False

    disc_metrics_real = discriminator.train_on_batch(real_data_X,real_data_Y)  
    disc_metrics_fake = discriminator.train_on_batch(fake_data_X,fake_data_Y)   
    
    if step%500 == 0:
      print("Disc: real loss: %f fake loss: %f" % (disc_metrics_real[0], disc_metrics_fake[0]))
    
    mean_disc_fake_loss.append(disc_metrics_fake[0])
    mean_disc_real_loss.append(disc_metrics_real[0])
    
    generator.trainable = True

    GAN_X = generate_noise(batch_size,noise_shape)
    GAN_Y = real_data_Y
    
    discriminator.trainable = False
    
    gan_metrics = gan.train_on_batch(GAN_X,GAN_Y)
    
    text_file = open(log_dir+"\\training_log.txt", "a")
    text_file.write("Step: %d Disc: real loss: %f fake loss: %f GAN loss: %f\n" % (tot_step, disc_metrics_real[0],
                                                                                   disc_metrics_fake[0],gan_metrics[0]))
    text_file.close()
    avg_GAN_loss.append(gan_metrics[0])
   
    if step%500==0:
      end_time = time.time()
      diff_time = int(end_time - begin_time)
      print("GAN loss: %f" % (gan_metrics[0]))
    
    if ((tot_step+1) % 500) == 0:
        print("-----------------------------------------------------------------")
        print("Average Discriminator_fake loss: %f" % (np.mean(mean_disc_fake_loss)))    
        print("Average Discriminator_real loss: %f" % (np.mean(mean_disc_real_loss)))    
        print("Average GAN loss: %f" % (np.mean(avg_GAN_loss)))
        print("-----------------------------------------------------------------")
        discriminator.trainable = True
        generator.trainable = True
        generator.save(generator_weights)
        discriminator.save(discriminator_weights)
