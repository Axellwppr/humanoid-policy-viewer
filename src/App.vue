<template>
  <v-app>
    <v-main>
      <Demo />
    </v-main>
  </v-app>
</template>

<script>
import Demo from '@/views/Demo.vue'

export default {
  name: 'App',
  components: {
    Demo
  }
}
</script>

<style>
body,
.v-application {
  font-family: 'Google Sans', 'Noto Sans SC', Arial, sans-serif !important;
}

html,
body {
  overflow-x: clip;
  width: 100%;
  position: relative;
}

.v-application {
  overflow-x: clip;
}

.v-container {
  max-width: 100%;
}
</style>
