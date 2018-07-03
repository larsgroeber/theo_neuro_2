# Import smtplib for the actual sending function
import smtplib

# Import the email modules we'll need
from email.mime.text import MIMEText
from time import sleep

mails = '''
larsgroeber7@gmail.com
keiwanjamaly@gmail.com
larsgroeber7@gmail.com
bastian@elearning.physik.uni-frankfurt.de
keiwan@elearning.physik.uni-frankfurt.de
elearning@itp.uni-frankfurt.de
'''.strip().split('\n')

textfile = './email.txt'


def connect():
    server = smtplib.SMTP('itp.uni-frankfurt.de', 587)
    server.connect('itp.uni-frankfurt.de', 587)
    server.ehlo()
    server.starttls()
    server.ehlo()
    # Next, log in to the server
    server.login("elearning", "POfb13kurzfilm")
    return server



# Open a plain text file for reading.  For this example, assume that
# the text file contains only ASCII characters.
with open(textfile) as fp:
    # Create a text/plain message
    mail_text = fp.read()

me = 'dsgvo@elearning.physik.uni-frankfurt.de'
for index, you in enumerate(mails):
    server = connect()
    msg = MIMEText(mail_text)
    msg['Subject'] = 'Information zum Thema Datenschutz und Verarbeitung von personenbezogenen Daten auf dem Elearning-Portal PhysikOnline'
    msg['From'] = me
    msg['To'] = you
    if msg['To'] != you:
        raise Exception(f'To header does not equal {you}!')
    #server.sendmail(me, you, msg.as_string())
    print(f'Send {index + 1}/{len(mails)} mail to {you}.')
    server.quit()
    sleep(1)
